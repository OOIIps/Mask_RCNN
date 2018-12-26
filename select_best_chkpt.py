import argparse
import tensorflow as tf
import glob
import os
import numpy as np
import json
import keras.backend as K
from config import Config
import model as modellib
import cv2

IOU_THRESHOLD = 0.99
slack = 1e-10

def iou_filter(bboxes, scores, cls, masks, iou_threshold = 0.5):
    
    def intersects(r1, r2):
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    def area(a, b):  # returns None if rectangles don't intersect
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx>=0) and (dy>=0):
            return dx*dy

    def union(a,b):
        return (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])       
    
    rboxes, rscores, rlabels, rmasks, ignore, inter = [], [], [], [], [], False
    for i in range(len(bboxes)):
        if i in ignore:
            continue
        for j in range(i+1, len(bboxes)):
            if intersects(bboxes[i], bboxes[j]):
                iou = area(bboxes[i], bboxes[j])/float(union(bboxes[i], bboxes[j])-area(bboxes[i], bboxes[j])+slack)
                if scores[i]<scores[j] and iou>iou_threshold:
                    inter = True
                elif scores[i]>scores[j] and iou>iou_threshold:
                    ignore.append(j)
            
        if not inter:                
            rboxes.append(bboxes[i])  
            rscores.append(scores[i])
            rlabels.append(cls[i])
            rmasks.append(masks[i])
        else:
            inter = False
    return rboxes, rscores, rlabels, rmasks

class Average:
    
    def __init__(self):
        self.hist = []
    
    def add(self, x):
        self.hist.append(x)

    def get(self):
        return np.mean(self.hist)

def IOU(box1, box2):
# (y1, x1, y2, x2)

    yc = max(box1[0], box2[0])
    xc = max(box1[1], box2[1])
    Yc = min(box1[2], box2[2])
    Xc = min(box1[3], box2[3])

    intersection_area = (Yc-yc)*(Xc-xc)

    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[1])*(box2[3]-box2[1])

    return intersection_area / (box1_area + box2_area - intersection_area + slack)

def evaluate_chkpt(modelpath, jsonfile, eval_type="bbox", num_images=100):
 
    with open(jsonfile, 'r') as f:
        jsonobj = json.load(f)

    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME='evaluation'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        
        def __init__(self, n):
            self.NUM_CLASSES = 1 + n 
            super().__init__()
    config = InferenceConfig(len(jsonobj['classes']))
    
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
   
    print("Loading weights ", modelpath)
    if 'mask_rcnn_coco' in modelpath.lower():
        model.load_weights(modelpath, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(modelpath, by_name=True)   
    
    AP, AR = Average(), Average()

    img_ct = len(jsonobj['images'])

    if img_ct < num_images:
        jsonobj['images'] = jsonobj['images']*(num_images//img_ct + 4)
    
    jsonobj['images'] = jsonobj['images'][:num_images]

    results = []
    for img in jsonobj['images']:
        # Load image
        image = cv2.imread(img['file_name'])
        
        anns = [ann for ann in jsonobj['annotations'] if ann['image_id']==img['id']]
        
        # Run detection
        r = model.detect([image], verbose=0)[0]
        
        bboxes, scores, classes, masks = r['rois'], r['class_ids'], r['scores'], r['masks']
        
        bboxes, scores, classes, masks = iou_filter(bboxes, scores, classes, masks)
        
        tp, fp, fn, gt_mark = 0, 0, 0, [False for _ in range(len(anns))]
        
        bboxes = [[b[1], b[0], b[3], b[2]] for b in bboxes]
            
        for i, (b, s, m) in enumerate(zip(bboxes, scores, masks)):
            marked = False
            for j, ann in enumerate(anns):
                if IOU(ann['bbox'], b) > IOU_THRESHOLD:
                    tp += 1
                    
                    '''
                    s = np.int32(ann['bbox'])
                    cv2.rectangle(image, (s[0], s[1]), (s[2], s[3]), (0,0,0), 2)
                    s = np.int32(b)
                    cv2.rectangle(image, (s[0], s[1]), (s[2], s[3]), (0,255,200), 2)
                    
                    cv2.imshow('f', image)
                    cv2.waitKey()
                    '''

                    marked = True
                    gt_mark[j] = True


            if not marked:
                fp += 1
        tn = len([x for x in gt_mark if not x])
        AP.add(tp/float(tp+fp+slack))
        AR.add(tp/float(tp+fn+slack))

    print ("[ Model %s ] Bounding Box AP: %f, AR: %f"%(modelpath, AP.get(), AR.get()))
        
def get_stats_from_file(eventfiles, metric="loss"):
    
    mrcnn_cls_losses, prev_steps = [], []
    for eventfile in tfevents:
        for idx, e in enumerate(tf.train.summary_iterator(eventfile)):
            for v in e.summary.value:
                if v.tag == metric:
                    
                    try:
                        prev_idx = prev_steps.index(e.step)
                        mrcnn_cls_losses[prev_idx][1] = np.mean([mrcnn_cls_losses[prev_idx][1], v.simple_value]) 
                    except ValueError:
                        mrcnn_cls_losses.append([e.step, v.simple_value])
                        prev_steps.append(e.step)
    return mrcnn_cls_losses

ap = argparse.ArgumentParser()
ap.add_argument('logdir', help='Path to checkpoint summary file(s) directory used for checkpointing')
ap.add_argument('jsonfile', help='Path to JSON dataset for validation')
ap.add_argument('--num_images', default=500, type=int, help='Path to JSON dataset for validation')
args = ap.parse_args()

tfevents = sorted(glob.glob(os.path.join(args.logdir, '*events.out.tfevents*')))
weights = sorted(glob.glob(os.path.join(args.logdir, '*.h5')))

metric_vals = get_stats_from_file(tfevents)
print ("\n".join(["step: %d, metric: %f"%(x[0], x[1]) for x in metric_vals]))

for wfile in weights:
    evaluate_chkpt(wfile, args.jsonfile, num_images=args.num_images)
