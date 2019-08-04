# coding: utf-8
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse
import json
import imutils
import glob

from .coco import BagsConfig

from . import model as modellib
from . import visualize_cv2 as visualize
#import visualize

def iou_filter(bboxes, cls, scores, iou_threshold = 0.5):

        def intersects(r1, r2):
            return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

        def area(a, b):  # returns None if rectangles don't intersect
            dx = min(a[2], b[2]) - max(a[0], b[0])
            dy = min(a[3], b[3]) - max(a[1], b[1])
            if (dx>=0) and (dy>=0):
                return dx*dy

        def union(a,b):
            return (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])

        rboxes, rscores, rlabels, ignore, inter = [], [], [], set(), False
        for i in range(len(bboxes)):
            if i in ignore:
                continue
            for j in range(i+1, len(bboxes)):
                if intersects(bboxes[i], bboxes[j]):
                    iou = area(bboxes[i], bboxes[j])/float(union(bboxes[i], bboxes[j])-area(bboxes[i], bboxes[j]))
                    if scores[i]<scores[j] and iou>iou_threshold:
                        inter = True
                    elif scores[i]>scores[j] and iou>iou_threshold:
                        ignore.add(j)

            if not inter:
                rboxes.append(bboxes[i])
                rscores.append(scores[i])
                rlabels.append(cls[i])
            else:
                inter = False
        return rboxes, rlabels, rscores

def inter_class_NMS(bboxes, cls, scores, masks, threshold=0.7):
        
    N = bboxes.shape[0]
    
    selected = []
    
    for i in range(N):
        
        ymin1, xmin1, ymax1, xmax1 = bboxes[i]

        rectangle = np.ones((xmax1-xmin1, ymax1-ymin1))
        
        total_area = np.sum(rectangle)

        for j in range(N):
            if i==j:
                continue
            else:
                ymin2, xmin2, ymax2, xmax2 = bboxes[j]
                
                if overlap(bboxes[i], bboxes[j]):
                    xo, yo, Xo, Yo = max([xmin1, xmin2]), max([ymin1,ymin2]), min([xmax1,xmax2]), min([ymax1,ymax2])
                    
                    # Normalize coordinates for overlap rectangle
                    xo -= xmin1
                    yo -= ymin1
                    Xo -= xmin1
                    Yo -= ymin1

                    # Eliminate area of overlap from area calculation
                    rectangle[xo:Xo, yo:Yo] = 0
      
        non_intersect_area = np.sum(rectangle)
        intersect_area = total_area - non_intersect_area
        
        iom = float(intersect_area) / (non_intersect_area + 1e-10)

        if iom < threshold:
            selected.append(i)
    
    if masks is not None:
        return bboxes[selected], cls[selected], scores[selected], masks[:,:,selected]
    else:
        return bboxes[selected], cls[selected], scores[selected], None

# Find if 2 rectangles overlap using SEPARATING AXIS THEOREM
def overlap(box1, box2):

    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    if xmin1 > xmax2 or xmin2 > xmax1:
        return False

    if ymin1 > ymax2 or ymin2 > ymax1:
        return False

    return True

class DemoConfig(BagsConfig):
    
    def __init__(self, n):
        self.NAME = 'bags'
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.NUM_CLASSES = 1 + n  # background + classes
        super().__init__(n)
    
if __name__=='__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Demo Mask R-CNN.')
    parser.add_argument('--json_file', required=True,
                        metavar="/path/to/json_file/",
                        help='Path to JSON file')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--rotation', required=False,
                        help="Angle to rotate video input stream")
    parser.add_argument('--video', required=False,
                        metavar="path/to/demo/video",
                        help='Video to play demo on', default=None)
    parser.add_argument('--image', required=False,
                        metavar="path/to/demo/image",
                        help='Video to play demo on')
    parser.add_argument('--image_dir', required=False,
                        metavar="path/to/demo/image/dir",
                        help='Image dir to play demo on')
    parser.add_argument('--show_upc', required=False, default=None,
                        help='Display UPC numbers instead of class_names from file')
    parser.add_argument('--save_demo', required=False, 
                        action='store_true', 
                        help='Saves demo to file instead of display')
    parser.add_argument('--resize', required=False,
                           help='Resizes demo for display by given fraction (0-1 input)')
    parser.add_argument('--loadconfig', required=False,
                           help='Loads configuration from file if provided', default=None)
    parser.add_argument('--disable_masks', action='store_true',
                           help='Loads configuration from file if provided', default=None)
    args = parser.parse_args()
    
    #cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    
    with open(args.json_file, 'r') as f:
        obj = json.load(f)
        
    config = DemoConfig(len(obj['classes']))
    config.BATCH_SIZE = 1
    config.MASK_LAYER = not args.disable_masks

    #config.from_json('demo_json.json')
    
    if args.loadconfig is not None:
        config.from_json(args.loadconfig)

    config.display()
    #exit()

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = args.model
    
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = ['000'] + json.load(open(args.json_file))['classes']
    if hasattr(args, 'upc_file'):
        with open(args.upc_file, 'r') as f:
            for line in f.readlines():
                code, cls = [x.strip() for x in line.split(' ')]
                class_names[class_names.index(cls)] = code
    
    if args.video is not None:
        ret, image = cap.read()
        if args.resize is not None:
            image = cv2.resize(image, (0,0), fx=float(args.resize), fy=float(args.resize))
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (image.shape[1], image.shape[0]))
    elif args.image_dir is not None:
        img_files=glob.glob(os.path.join(args.image_dir, '*'))
        ret = True
    else:
        image = cv2.imread(args.image)
        ret = False      
        out=None
    ind = 0
    while (1):
        if args.video is not None:
            ret, image = cap.read()
        elif args.image_dir is not None:
            if ind == len(img_files):
                break
            image = cv2.imread(img_files[ind])
            ind+=1
            out = None
        else:
            image = cv2.imread(args.image)
            ret = not ret    
        if args.resize is not None:
            image = cv2.resize(image, (0,0), fx=float(args.resize), fy=float(args.resize))
        if not ret:
            break
        
        if args.rotation is not None:
            image = imutils.rotate_bound(image, int(args.rotation))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        
        r['rois'], r['class_ids'], r['scores'] = [np.array(x) for x in iou_filter(r['rois'], r['class_ids'], r['scores'])]

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], save=args.save_demo)
        
        r['rois'], r['class_ids'], r['scores'], r['masks'] = \
                inter_class_NMS(r['rois'], r['class_ids'], r['scores'], r['masks'])        

        t = 'image' if args.image is not None else 'video'
       
        '''
        #print (r['rois'], r['class_ids'])
        
        threshold = 75
        
        h_prev, h_prev2, remove = None, None, []
        for i,x in enumerate(r['class_ids']):
            if x==17: # Rhythm journal
                s = r['rois'][i]
                h,w = s[2]-s[0], s[3]-s[1]
                if h_prev is None:
                    h_prev, w_prev, i_prev = h,w,i
                    
                if abs(h-h_prev) < threshold and abs(w-w_prev) < threshold:
                    continue
                else:
                    if h_prev2 is None:
                        h_prev2, w_prev2, i_prev2 = h, w, i 
                    else:                   
                        if abs(h-h_prev2) < threshold and abs(w-w_prev2) < threshold:
                            h_prev, w_prev, i_prev = h_prev2, w_prev2, i_prev2
                        else:
                            remove.append(i)
        for r in list(reversed(remove)):
            print ("Deleted row ", r)
            r['class_ids'] = np.delete(r['class_ids'], (r), axis=0)
            r['rois'] = np.delete(r['rois'], (r), axis=0)
            r['masks'] = np.delete(r['masks'], (r), axis=0)
            r['scores'] = np.delete(r['scores'], (r), axis=0)
        
        if h_prev2 is not None:
            print ("Deleted row ", i_prev2)
            r['class_ids'] = np.delete(r['class_ids'], (i_prev2), axis=0)
            r['rois'] = np.delete(r['rois'], (i_prev2), axis=0)
            r['masks'] = np.delete(r['masks'], (i_prev2), axis=0)
            r['scores'] = np.delete(r['scores'], (i_prev2), axis=0)
        '''

        if not args.save_demo:
            c = cv2.waitKey()
            if args.image_dir is not None:
                if c==81:
                    ind-=2
    if args.video is not None:
        cap.release()
        out.release()
    cv2.destroyAllWindows()
