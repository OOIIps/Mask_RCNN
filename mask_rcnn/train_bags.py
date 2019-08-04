import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
import xml.etree.ElementTree as ET

from .rgb_segmentor import get_rgb_masks
from .hsv_segmentor import get_hsv_masks

from .config import Config
import .utils
import .model as modellib
import .visualize
from .model import log

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on Custom Bags Dataset.')
parser.add_argument("command", metavar="<command>", help="'train' or 'eval'")
parser.add_argument('--model', required=False, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
parser.add_argument('--logs', required=False, default='log/', metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
args = parser.parse_args()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class BagsConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "bags"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12  # background [index: 0] + 12 classes
    STEPS_PER_EPOCH = 3000
    VALIDATION_STEPS = 100
    
config = BagsConfig()
config.display()

class InferenceConfig(BagsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class BagsDataset(utils.Dataset):
    """Generates the bags dataset. 
    """
    
    def load_bags(self, part):
        """
        part: train/eval
        """

        classes = ['black_backpack', 'nine_west_bag', 'meixuan_brown_handbag', 'sm_bdrew_grey_handbag', 'wine_red_handbag', 'sm_bclarre_blush_crossbody', 'mk_brown_wrislet', 'black_plain_bag', 'lmk_brown_messenger_bag', 'sm_peach_backpack', 'black_ameligalanti', 'white_bag']
        
        count = 1

        # Add classes
        
        for i, c in enumerate(classes):
            self.add_class("bags", i+1, c)
        
        # Add train/val images
        
        pattern = re.compile(".*bot[0-9]*.png")
        
        for images in glob.glob(os.getcwd()+'/Data/handbag_images/JPEGImages/*.png'):
            
            f = images.split('JPEGImages')
            ann_path = f[0]+'Annotations'+f[1][:-3]+'xml'
            
            tree = ET.parse(ann_path)
            root = tree.getroot()         
            width, height = int(root.find('size').find('width').text), int(root.find('size').find('height').text)
            
            if height>config.IMAGE_MAX_DIM or width>config.IMAGE_MAX_DIM or height<config.IMAGE_MIN_DIM or width<config.IMAGE_MIN_DIM:
                continue
            
            if(pattern.match(images.split('/')[-1]) and part=='eval'):
                self.add_image('bags', image_id = count, path = images, width=width, height=height)
                
            if(not pattern.match(images.split('/')[-1]) and part=='train'):
                self.add_image('bags', image_id = count, path = images, width=width, height=height)
            
            count+=1
        '''
        if (part == 'train'):
            for class_path in glob.glob('Data/bags/*.jpg'):
                for file_path in glob.glob(class_path+'/*'):
                    img = cv2.imread(file_path).shape
                    segments, bboxes = find_object_bbox_masks(file_path)
                    
                    self.add_image('bags', image_id = count, path = file_path, width=img[1], height=img[0], bags=[[class_path.split('/')[-1], segments[:,:,0], bboxes[:,0]]])
                    count+=1       
        '''

    def image_reference(self, image_id):
        """Return the bags data of the image."""
        info = self.image_info[image_id]
        return info["path"]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        image_masks, classes = get_hsv_masks(info['path'])
        class_ids = np.array([self.class_names.index(s) for s in classes])
        return image_masks, class_ids.astype(np.int32)

# Training dataset
dataset_train = BagsDataset()
dataset_train.load_bags('train')
dataset_train.prepare()

print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))

dataset_val = BagsDataset()
dataset_val.load_bags('eval')
dataset_val.prepare()

model = None

if (args.command!='train'):
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
else:
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "imagenet"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

if (args.command=='train'):

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=10,
                layers='all')

elif args.command == "eval":
    
    # Validation dataset
    
    evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    image_ids = random.choice(dataset_val.image_ids, 10)
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
