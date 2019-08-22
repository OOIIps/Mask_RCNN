import os
import time
import numpy as np

from mask_rcnn.parallel_model import ParallelModel
from mask_rcnn.demo import iou_filter
from mask_rcnn import visualize_cv2 as visualize

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

from PIL import Image
import rawpy

from imgaug import augmenters as iaa

from mask_rcnn.config import Config
from mask_rcnn import utils
from mask_rcnn import model as modellib

from skimage.measure import find_contours as FC
from simplification.cutil import simplify_coords

DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), "logs")

FISH_PARTS = ['anal_fin', 'caudal_fin', 'dorsal_fin', 'dorsal_side', 'eye', 'head', 'humeral_blotch', \
                'operculum', 'original',  'pectoral_fin', 'pelvic_fin', 'stripes', 'ventral_side', 'whole_body']

IMG_TYPES = ['jpg', 'png', 'arw']
IMG_TYPES.extend([x.upper() for x in IMG_TYPES])

def gamma_correction(image, gamma=1.0):
   inv = 1.0 / gamma
   table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

class FishDataset(utils.Dataset):
    def __init__(self, split='train'):
        
        super().__init__()

        self.name = 'fish'
        
        if split == 'train':
            self.split_ratio = 0.9
        else:
            self.split_ratio = 0.1
        
        self.split = split

    def load_data(self, folders, dtype, name=None):
        
        assert dtype in FISH_PARTS

        self.dtype = dtype
        
        self.img_files, self.ann_files = {}, {}

        for folder in folders:
            
            self.img_files.update(self.get_image_files(folder))
            
            dpath = os.path.join(folder, dtype)
            self.ann_files.update(self.get_ann_files(dpath, self.img_files))
        
        L = sorted(list(self.img_files.keys()))
        N = len(L)
        n = int(N * self.split_ratio)
        
        if self.split == 'train':
            b,e = n+1, N
        else:
            b,e = 0, N-n

        for idx in range(e-1, b-1, -1):
            del self.img_files[L[idx]]
            del self.ann_files[L[idx]]
            del L[idx]
        
        self.ordered_keys = L
        
        self.mean, self.std = self.calculate_rough_image_stats()

        # Add images
        for idx, key in enumerate(self.ordered_keys):
            
            try:
                img = cv2.imread(self.img_files[key])
                mask = cv2.imread(self.ann_files[key])

                if img is None or mask is None:
                    continue

            except Exception:
                continue
            
            self.add_image(
                    self.name, image_id=idx,
                    path=os.path.abspath(self.img_files[key]),
                    width=img.shape[1],
                    height=img.shape[0],
                    annotations=[self.ann_files[key]])
                #print (self.image_info, self.image_ids)
        
        # Add class
        self.add_class(self.name, 1, self.dtype)

        print ('Using %s dataset of %d/%d total images'%(self.split, len(self.image_info), len(self.ordered_keys)))

    def get_image_files(self, path):
        
        imgs = [y for x in [glob.glob(os.path.join(path, '*.'+e)) 
                  for e in IMG_TYPES] for y in x]

        img_dict = {}
        for path in imgs:
            sfx = '.'.join(path.split('/')[-1].split('.')[:-1])
            img_dict[sfx] = path

        return img_dict
    
    def calculate_rough_image_stats(self):
        
        stat_files = glob.glob('cache/*_count.txt')

        if len(stat_files) >= 1:
            for idx, f in enumerate(stat_files):
                with open(f, 'r') as g:
                    if idx == 0:
                        ctx = int(g.read().strip())
                        max_f = f
                    elif int(g.read().strip()) > ctx:
                        ctx = int(g.read().strip())
                        max_f = f
        
            if ctx > len(self):
                with open(max_f.replace('count', 'stats'), 'r') as f:
                    mean, std = [np.array([np.float(y) for y in x.strip().split(' ')]) 
                                        for x in f.readlines()]
                    return mean, std
                
        mean = np.zeros(3)
        std = np.zeros(3)
        for key in self.img_files:
            try:
                img = np.array(self.get_image(self.img_files[key]))
            except Exception:
                continue

            mean += np.mean(np.mean(img, axis=0), axis=0)
            std += np.std(np.std(img, axis=0), axis=0)
            
        nb_samples = len(self.img_files)
        mean /= nb_samples
        std /= nb_samples

        if not os.path.isdir('cache'):
            os.mkdir('cache')

        with open('cache/%s_stats.txt'%(self.split), 'w') as f:
            f.write(' '.join(['%.5f'%(x) for x in mean])+'\n')
            f.write(' '.join(['%.5f'%(x) for x in std]))
        
        with open('cache/%s_count.txt'%(self.split), 'w') as g:
            g.write('%d'%len(self))

        return mean, std

    def get_segmentation_mask(self, path):
        
        ANN = self.get_image(path)        
        gray = ANN.convert('L')
        bw = gray.point(lambda x: 0 if x==255 else 1, '1')
        
        #bw.save('sample.jpg')
        #print (np.max(np.array(bw))) 
        #print (np.min(np.array(bw))) 
        
        return bw
    
    def get_ann_files(self, path, imgs):
        
        ann_dict = {}
        for img_key in imgs:
            
            annfile = glob.glob(os.path.join(path, '*'+img_key+' *'))
            annfile.extend(glob.glob(os.path.join(path, '*'+img_key+'.*')))
            assert len(annfile) <= 1
            if annfile:
                ann_dict[img_key] = annfile[0]
                
                #img = self.get_image(imgs[img_key])
                #img.save('sample2.jpg')
                
        return ann_dict
    
    def __len__(self):
        return len(self.img_files)

    def get_image(self, path):
        
        if 'arw' in path.lower():
            raw = rawpy.imread(path)
            img = raw.postprocess()
            img = Image.fromarray(img)
        else:
            img = Image.open(path)
        
        if len(np.array(img).shape) > 2 and np.array(img).shape[2] == 4:
            img = np.array(img)[:,:,:3]
            img = Image.fromarray(img)
        return img
    
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
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = 1 # one object detection problem

            ann = cv2.imread(annotation)
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2GRAY)
            #mask = cv2.resize(mask, (image_info['width'], image_info['height']))
            ann[ann==255]=0
            ann[ann>0]=1

            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if ann.max() < 1:
                continue
            
            '''
            #debug masks
            img = cv2.imread(image_info['path'])
            ann = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR)
            added_image = cv2.addWeighted(img,0.4,ann,0.5,0)
        
            cv2.imshow('g', added_image)
            cv2.waitKey()
            '''

            instance_masks.append(ann)
            class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FishDataset, self).load_mask(image_id)

############################################################
#  Training
############################################################

class FishConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "fish"

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
    RPN_ANCHOR_SCALES = (32, 64, 128, 192, 256) 
    RPN_ANCHOR_RATIOS = [0.3, 0.6, 1.2] #[0.5, 1, 2] height/width - 0.5 means wide
    USE_MINI_MASK = True
    MAX_GT_INSTANCES = 100

    def __init__(self, n, m=1, steps=500, name='fish', mask_layer=True, test=False):
        self.NAME = name
        self.NUM_CLASSES = 1 + n 

        if not test:
            self.GPU_COUNT = m
            self.STEPS_PER_EPOCH = steps
        else:
            self.GPU_COUNT = 1
            self.IMAGES_PER_GPU = 1

        self.MASK_LAYER = mask_layer
        super().__init__()

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
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
    parser.add_argument('--num_gpus', default=3, type=int, help='Number of GPUs available')
    parser.add_argument('--nsteps', default=500, type=int, help='Number of steps per epoch')
    parser.add_argument('--project', default='fish', help='Name of project for checkpointing')
    parser.add_argument('--loadconfig', default=None, help='Path to config file for model')
    parser.add_argument('--ignore_masks', action='store_true', help='Set this flag to ignore mask layer during training and inference')
    parser.add_argument('--save_demo', action='store_true', help='Flag to save images')
    parser.add_argument('--folders', nargs='+', help='Path to folders with fish data images', required=True)
    parser.add_argument('command', choices=['train', 'test'], help='Command deciding between inference and training (train/test)')
    args = parser.parse_args()
    
    config = FishConfig(1, args.num_gpus, args.nsteps, args.project, not args.ignore_masks, args.command == 'test')
            
    if args.loadconfig is not None:
        config.from_json(args.loadconfig, train = (args.command == 'train'))
    
    print("Loading weights ", args.model)
    
    if args.command == 'train':

        aug = iaa.OneOf(
                       [iaa.Flipud(0.5), iaa.Fliplr(0.5), 
                        iaa.GaussianBlur(sigma=(0.0, 3.0)),
                        iaa.AdditiveGaussianNoise(scale=0.05*255), 
                        iaa.ElasticTransformation(sigma=5.0), 
                        iaa.ContrastNormalization((0.5, 1.5)),
                        iaa.Affine(translate_percent={'x':(-0.2, 0.2), 'y': (-0.2, 0.2)}), 
                        iaa.Affine(rotate=(-30, 30)),
                        iaa.CropAndPad(percent=(-0.25, 0.25))])
    
        dataset_train = FishDataset()
        dataset_train.load_data(args.folders, 'whole_body', name=args.project)
        dataset_train.prepare()
        
        # Validation dataset
        dataset_val = FishDataset('val')
        dataset_val.load_data(args.folders, 'whole_body', name=args.project)
        dataset_val.prepare()
        
        config.MEAN_PIXEL = dataset_train.mean
        config.display()
    
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
        
        if 'mask_rcnn' in args.model.lower() and 'logs' not in args.model.lower():
            model.load_weights(args.model, by_name=True, \
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(args.model, by_name=True)
        
        schedfactor = 5
        lr = 1e-3

        schedule = lambda epoch: lr*np.power(0.15, epoch // schedfactor)

        temps = 40
        print("Training network heads on augmented dataset")
        model.train(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=temps,
                    layers='heads',
                    augment=True,
                    augmentation=aug,
                    lr_decay_schedule=schedule)
        
        schedule = lambda epoch: lr*np.power(0.35, (epoch-temps) // schedfactor)
        
        temps1 = temps + 30
        print("Training network heads on full dataset")
        model.train(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=temps1,
                    layers='heads',
                    augment=False,
                    augmentation=None,
                    lr_decay_schedule=schedule)

        lr /= 10
        schedule = lambda epoch: lr*np.power(0.15, (epoch - temps1) // schedfactor)
        
        temps2 = temps1 + 40
        print("Fine tune all layers on augmented dataset")
        model.train(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=temps2,
                    layers='all',
                    augment=True,
                    augmentation=aug,
                    lr_decay_schedule=schedule)

        schedfactor = 3  
        schedule = lambda epoch: lr*np.power(0.1, (epoch-temps2) // schedfactor)
        
        temps3 = temps2 + 15
        print("Fine tune all layers on full dataset")
        model.train(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=temps3,
                    layers='all',
                    augment=False,
                    augmentation=None,
                    lr_decay_schedule=schedule)
    
    else:

        stat_files = glob.glob('cache/*_count.txt')

        if len(stat_files) >= 1:
            for idx, f in enumerate(stat_files):
                with open(f, 'r') as g:
                    if idx == 0:
                        ctx = int(g.read().strip())
                        max_f = f
                    elif int(g.read().strip()) > ctx:
                        ctx = int(g.read().strip())
                        max_f = f
        
            with open(max_f.replace('count', 'stats'), 'r') as f:
                mean, std = [np.array([np.float(y) for y in x.strip().split(' ')]) 
                                    for x in f.readlines()]
                config.MEAN_PIXEL = mean
        
        config.display()

        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        model.load_weights(args.model, by_name=True)
        
        img_files = glob.glob(os.path.join(args.folders[0], '*'))

        for path in img_files:
            image = cv2.imread(path)

            if image is not None:
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                results = model.detect([image], verbose=1)

                r = results[0]
        
                r['rois'], r['class_ids'], r['scores'] = [np.array(x) for x in iou_filter(r['rois'], r['class_ids'], r['scores'])]

                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            ['BG', args.project], r['scores'], save=args.save_demo)
        
                cv2.waitKey()
