"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np

from .parallel_model import ParallelModel

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/CocoDataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import json 
import zipfile
import urllib.request
from collections import defaultdict
import shutil
import json
import cv2
import glob
import itertools

from imgaug import augmenters as iaa

from .config import Config
from . import utils
from . import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

def gamma_correction(image, gamma=1.0):
   inv = 1.0 / gamma
   table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

############################################################
#  Dataset
############################################################

class TrainDataset(utils.Dataset):
    def load_data(self, json_path, name=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        
        NAME = "bags" if name is None else name 
        print (NAME)
        # Add classes
        for i, cls in enumerate(sorted(dataset['classes'])):
            self.add_class(NAME, i+1, cls)
        
        imgToAnns = defaultdict(list)
        for ann in dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
        
        self.max_img_id = -1
        # Add images
        for img_json in dataset['images']:
            i, img_path = img_json['id'], img_json['file_name']
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    self.add_image(
                        NAME, image_id=i,
                        path=os.path.abspath(img_path),
                        width=img.shape[1],
                        height=img.shape[0],
                        annotations=imgToAnns[i])
            
            if i > self.max_img_id:
                self.max_img_id = i
   
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        image_info = self.image_info[image_id]
        
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        #        dic2 = {'segmentation': poly, 'area': area, 'iscrowd':0, 'image_id':i, 'bbox':bbox, 'category_id': cat_id, 'id': ann_index}
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(TrainDataset, self).load_mask(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://CocoDataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

class BagsConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "lasa"

    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 100
    IMAGE_MIN_DIM = 768 #512#1024
    IMAGE_MAX_DIM = 1024 #512#1024
    IMAGE_PADDING = True
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = 0.33 #0.33
    MEAN_PIXEL = np.array([105.73722594, 103.80840869, 101.94847263])
    BACKBONE='resnet101'
    LEARNING_RATE = 1e-3
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 192) 
    RPN_ANCHOR_RATIOS = [0.5, 1.2, 3] #[0.5, 1, 2] height/width - 0.5 means wide
    USE_MINI_MASK = True
    MAX_GT_INSTANCES = 100

    def __init__(self, n, m=1, steps=500, name='wirecard', mask_layer=True):
        self.NAME = name
        self.NUM_CLASSES = 1 + n 
        self.GPU_COUNT = m
        self.STEPS_PER_EPOCH = steps
        self.MASK_LAYER = mask_layer
        super().__init__()

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--json_file', required=False,
                        metavar="/path/to/json_file/",
                        help='Path to JSON file')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--augment', required=False, action='store_true', help='add augmentations')
    parser.add_argument('--stage', required=True, default=1, type=int, help='Choose stage of training (1-heads, 2-4+, 3-full)')
    parser.add_argument('--num_gpus', default=3, type=int, help='Number of GPUs available')
    parser.add_argument('--nsteps', default=500, type=int, help='Number of steps per epoch')
    parser.add_argument('--project', default='wirecard', help='Name of project for checkpointing')
    parser.add_argument('--loadconfig', default=None, help='Path to config file for model')
    parser.add_argument('--ignore_masks', action='store_true', help='Set this flag to ignore mask layer during training and inference')
    args = parser.parse_args()
    
    if args.augment:
        aug = iaa.OneOf(
                   [iaa.Flipud(0.5), iaa.Fliplr(0.5), 
                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
                    iaa.AdditiveGaussianNoise(scale=0.05*255), 
                    iaa.ElasticTransformation(sigma=5.0), 
                    iaa.ContrastNormalization((0.5, 1.5)),
                    iaa.Affine(translate_percent={'x':(-0.2, 0.2), 'y': (-0.2, 0.2)}), 
                    iaa.Affine(rotate=(-30, 30)),
                    iaa.CropAndPad(percent=(-0.25, 0.25))])
        augment=True # GAMMA CORRECTION 
    else:
        aug = None
        augment=False # GAMMA CORRECTION
    ############################################################
    #  Configurations
    ############################################################

    with open(args.json_file, 'r') as f:
        obj = json.load(f)
    
    # Configurations
    if args.command == "train":
        config = BagsConfig(len(obj['classes']), args.num_gpus, args.nsteps, args.project, not args.ignore_masks)
    else:
        class InferenceConfig():
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.5
            DETECTION_NMS_THRESHOLD = 0.4
            
            def __init__(self, n):
                self.NUM_CLASSES = 1 + n 
                super().__init__()
        
        config = InferenceConfig(len(obj['classes']))
    
    if args.loadconfig is not None:
        config.from_json(args.loadconfig, train=True)
    
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
   
    # Load weights
    print("Loading weights ", args.model)
    if 'mask_rcnn' in args.model.lower() and 'logs' not in args.model.lower():
        model.load_weights(args.model, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(args.model, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = TrainDataset(len(obj['classes']))
        dataset_train.load_data(args.json_file, name=args.project)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = TrainDataset(len(obj['classes']))
        dataset_val.load_data(args.json_file, name=args.project)
        dataset_val.prepare()
        
        temps = 30
        # *** This training schedule is an example. Update to your needs ***
        if args.stage==1:     
            # Training - Stage 1
            print("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=0.001,
                        epochs=temps,
                        layers='heads',
                        augment=augment,
                        augmentation=aug)
        elif args.stage==2:
            # Training - Stage 2
            print("Training 4+ layers")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 10,
                        epochs=temps+40,
                        layers='4+',
                        augment=augment,
                        augmentation=aug)
        else:
            #Training - Stage 3
            # Fine tune all layers
            print("Fine tune all layers")
            model.train(dataset_train, dataset_val,
                        learning_rate=0.0001,
                        epochs=temps+40+15,
                        layers='all',
                        augment=augment,
                        augmentation=aug)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = TrainDataset()
        coco = dataset_val.load_data(args.json_file, name=args.project)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
