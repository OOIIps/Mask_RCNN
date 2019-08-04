from PIL import Image
import glob
import numpy as np
import cv2
import os
import math
from cv2 import moveWindow
from scipy import ndimage
from collections import Counter
import xml.etree.ElementTree as ET  
import time

from matplotlib import patches
import matplotlib.pyplot as plt

#fig = plt.figure()

def find_object_bbox_masks(file_name, shapes=None):

    calculate_bbox = True
    bboxes, segments = None, None
    if (shapes is not None):
        bboxes = [bbox for _, bbox in shapes]
        calculate_bbox = False
    else:
        shapes = [[file_name.split('/')[-2], [0, 0, Image.open(file_name).size[1], Image.open(file_name).size[0]]]]
    annot_path = file_name[:-4]
    
    if (os.path.exists(annot_path+'_mask.npy') and os.path.exists(annot_path+'_bbox.npy')):
        segments = np.load(annot_path+'_mask.npy')
        bboxes = np.load(annot_path+'_bbox.npy')
    else:
        im = Image.open(file_name)
        sorte = im.getcolors(im.size[0]*im.size[1])
        sorte.sort(reverse=True, key= lambda x: x[0])

        one = np.array(sorte[0][1])
        two = np.array(sorte[1][1])
        dif = sum(abs(one-two))
        
        for n, col in sorte:
            if (sum(abs(one-col))>dif):
                two = col
                dif = sum(abs(one-col))
        
        if(dif<65):
            return None
            
        if(sum(abs(one-np.array(im)[0][0]))<sum(abs(two-np.array(im)[0][0]))):
            temp = one
            one = two
            two = temp
        
        min_x, min_y, max_x, max_y = np.shape(np.array(im))[0], np.shape(np.array(im))[1], 0, 0
        segments, bboxes = np.zeros((im.size[0], im.size[1], len(shapes))), []
        
        for i, (target_class, bbox) in enumerate(shapes):
            
            if (max(bbox)<1):
                if(len(shapes)==1):
                    shapes[0][1] = [0, 0, im.size[1], im.size[0]]
                else:
                    continue
            
            for key1, vals in enumerate(np.array(im)[int(bbox[0]):int(bbox[1])]):
                for key2, rgb in enumerate(vals[int(bbox[2]):int(bbox[3])]):
                    if(sum(abs(rgb-one))<700):
                        segments[key1+int(bbox[0])][key2+int(bbox[2])][i]=1
                        if (calculate_bbox):
                            if(key1+int(bbox[1])<min_x):
                                min_x = key1+int(bbox[1])
                            if(key2+int(bbox[0])<min_y):
                                min_y = key2+int(bbox[0])
                            if(key1+int(bbox[3])>max_x):
                                max_x = key1+int(bbox[3])
                            if(key2+int(bbox[2])>max_y):
                                max_y = key2+int(bbox[2])
            if (calculate_bbox):
                if (max_x>=im.size[0] or max([min_x, min_y, max_x, max_y])<1):
                    max_x = im.size[0]-1
                if (max_y>=im.size[1] or max([min_x, min_y, max_x, max_y])<1):
                    max_y = im.size[1]-1
            
            if (calculate_bbox):
                bboxes.append([min_x, min_y, max_x, max_y])
        
        if(not calculate_bbox):
            bboxes =  np.array([bbox for name, bbox in shapes])
        
        np.save(annot_path+'_mask', np.array(segments, dtype=bool))
        np.save(annot_path+'_bbox', np.array(bboxes, dtype=np.int32))
        
    return np.array(segments, dtype=np.int32), np.array(bboxes, dtype=np.int32)

def return_mask(image_path, bbox, bag_type):
    
    '''
    image_path: Location to image file
    bbox: Numpy array of [ymin, xmin, ymax, xmax]
    bag_type: Class of Bbox
    
    Returns:
    mask: Image segmentation mask based on color contrast in height*width Numpy array
    '''
    orig_im = im = Image.open(image_path)
    im = Image.fromarray(np.array(im)[bbox[0]:bbox[2], bbox[1]:bbox[3]], 'RGB')
    
    #orig_im.show()
    #im.show()
    
    mask = np.zeros((im.size[1], im.size[0]), dtype=np.uint8)
    sorte = im.getcolors(im.size[0]*im.size[1])
    sorte.sort(reverse=True, key= lambda x: x[0])

    one = np.array(sorte[0][1])
    two = np.array(sorte[1][1])
    dif = sum(abs(one-two))
    
    for n, col in sorte:
        if (sum(abs(one-col))>dif):
            two = col
            dif = sum(abs(one-col))
    
    if(sum(abs(one-np.array(im)[0][0]))<sum(abs(two-np.array(im)[0][0]))):
        temp = one
        one = two
        two = temp
    
    for key1, vals in enumerate(np.array(im)):
        for key2, rgb in enumerate(vals):
            if (key1<mask.shape[0] and key2<mask.shape[1]):
                if(sum(abs(rgb-one))<575 and 'black' not in bag_type):
                    mask[key1][key2]=255
                elif(sum(abs(rgb-one))>575 and 'black' in bag_type):
                    mask[key1][key2]=255
    
    #print (np.count_nonzero(mask)/(im.size[0]*im.size[1]))
    if ((np.count_nonzero(mask)/(orig_im.size[0]*orig_im.size[1]))<0.025):
        mask = np.ones((im.size[1], im.size[0]), dtype=np.uint8)*255
    
    #print (mask.shape)
    height, width = orig_im.size[:2]
    #print (height, width)
    
    #Image.fromarray(mask, 'L').show()
    
    mask = np.lib.pad(mask, ((bbox[0],width-bbox[2]), (bbox[1],height-bbox[3])), 'constant', constant_values=0)
    
    return mask    

def get_image_masks(image_path):
    
    split = image_path.split('JPEGImages')
    
    annot_path = split[0]+'Annotations'+split[1][:-3]+'xml'
    
    #print (annot_path)
    
    tree = ET.parse(annot_path)
    root = tree.getroot()         
    
    num_masks, width, height = len(root.findall('object')), int(root.find('size').find('width').text), int(root.find('size').find('height').text)
    
    image_masks, classes = np.zeros((height, width, num_masks), dtype=np.uint8), []
    
    for i, obj in enumerate(root.findall('object')):
    
        cls = obj.find('name').text
        bx = [int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymax').text), int(obj.find('bndbox').find('xmax').text)]
        image_masks[:,:,i] = return_mask(image_path, np.array(bx), cls)
        classes.append(cls)
    
    orig_im = Image.open(image_path)
    
    for i in range(image_masks.shape[-1]):
        mask = Image.fromarray(np.dstack((image_masks[...,i], image_masks[...,i], image_masks[...,i])), 'RGB')
        Image.blend(orig_im, mask, 0.75).show()
    
    return image_masks, classes

def test_time():
    tic = time.time()
    
    for f in glob.glob(os.getcwd()+'/Data/handbag_images/JPEGImages/*.png'):
        get_image_masks(f)

    #seg, bb = find_object_bbox_masks(os.getcwd()+'/Data/bags/black_ameligalanti/2017-L7-CK2-20780452-01-1.jpg')

    #Image.fromarray(return_mask('/home/hans/Desktop/Vision Internship/github/Mask_RCNN/Data/handbag_images/JPEGImages/bot3.png', np.array([1, 1, 178, 192]), 'black_ameligalanti'), 'L').show()

    print (time.time()-tic)

test_time()
