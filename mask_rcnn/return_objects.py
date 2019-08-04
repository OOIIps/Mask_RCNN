import os
import sys
import random
import math
import cv2
import json

from .config import Config

import .model as modellib

class BagsConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bags"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12  # background [index: 0] + 1 person class tranfer from COCO [index: 1] + 12 classes

class Hans1:

    def __init__(self, json_file, model_path):
        config = BagsConfig()

        self.model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(model_path), config=config)
        self.model.load_weights(model_path, by_name=True)

        self.class_names = [ 'BG' ] + json.load(open(json_file))['classes']


    def return_objects(self, image):
        # ## Run Object Detection
        labels = self.model.detect([image], verbose=0)

        labels = labels[0]

        results = []
        for i in range(len(labels['class_ids'])):
            results.append({
                'class_idx': labels['class_ids'][i],
                'class_name': self.class_names[labels['class_ids'][i]],
                'score': labels['scores'][i],
                'roi': labels['rois'][i],
                'mask': labels['masks'][i]
            })

        return results

if __name__ == '__main__':
    ai = Hans1('model/CISCO/cisco_new.json', 'model/CISCO/cisco_mask_rcnn_csv.h5')
    #print(ai.return_objects('1top1.png'))
