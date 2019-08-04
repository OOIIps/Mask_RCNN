
# coding: utf-8

# In[2]:

import time
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import scipy
import json

import .coco
import .utils
import .model as modellib
import .visualize

import sys, time
sys.path.append('/home/hans/libfreenect/wrappers/python/')
import freenect
import cv2
import frame_convert2

from .config import Config

import argparse

from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
#rcParams['figure.axes'] = False

parser = argparse.ArgumentParser(description='Bounding Box Detection or Image Segmentation')
parser.add_argument("--bbox", default=False, action="store_true" , help="Set for Bounding Box Detection")
parser.add_argument("--model", required=True, help='Path to model file')
parser.add_argument("--json_file", required=True, help='Path to dataset JSON file')
args = parser.parse_args()


# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = args.model

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:

with open(args.json_file, 'r') as f:
    obj = json.load(f)

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME='bags'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    def __init__(self, n):
        self.NUM_CLASSES = 1 + n
        super().__init__()

config = InferenceConfig(len(obj['classes']))
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(args.model), config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# In[5]:

def apply_mask(image, mask, color, alpha=0.25):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def get_depth():
    return freenect.sync_get_depth()[0]

vid = cv2.VideoCapture(0)

def get_video():
    ret, image = vid.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #random.shuffle(colors)
    return colors

starttime = time.time()
TOTAL = 0

patches = []
fig, ax = plt.subplots(1)

image_actor, text_actors = ax.imshow(scipy.misc.imresize(get_video(), (256, 256)).astype(np.uint8), animated=True), [ax.text(0, 0, "", color='w', size=11, backgroundcolor="black") for _ in range(100)]

def animate(*kwargs):
    # Load a random image from the images folder
    global TOTAL, patches
    for p in patches:
        p.remove()
    patches = []
    
    ax.cla()
    TOTAL+=1
    plt.title("Bounding Box Detection" if args.bbox else "Image Segmentation")
    plt.axis("off")
    
    image = scipy.misc.imresize(get_video(), (256, 256))
    #file_names = next(os.walk(IMAGE_DIR))[2]
    
    
    
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # Run detection
    temps = time.time()
    results = model.detect([image], verbose=1)
    print("TIME FOR SEGMENTATION: ", time.time()-temps, "sec")
    # Visualize results
    r = results[0]
    boxes, masks, class_ids, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                            class_names, r['scores'], bbox=False)
    
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7,
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        patches.append(p)
    
        if not args.bbox:
            # Mask
            mask = masks[:, :, i]
            image = apply_mask(image, mask, color)
            y1, x1, y2, x2 = boxes[i]
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
                patches.append(p)
                    
        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        text_actors[i].set_x(x1)
        text_actors[i].set_y(y1+8)
        text_actors[i].set_text(caption)
        
        
    image_actor.set_data(image.astype(np.uint8))
    
    return [image_actor] + text_actors[:N]
    
ani = animation.FuncAnimation(fig, animate, blit=True)

plt.show()

print ( 'FPS:', TOTAL / ( time.time() - starttime ) )
