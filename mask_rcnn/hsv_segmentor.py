import cv2
import numpy as np 
import glob
import os
from PIL import Image
import time
import xml.etree.ElementTree as ET  
from rgb_segmentor import return_rgb_mask

def return_hsv_mask(image_path, bx, cls):

    frame = cv2.imread(image_path) 
    height, width = frame.shape[:2]
    
    #print (bx)
    #print (height, width)
    
    frame = frame[bx[0]:bx[2], bx[1]:bx[3]]
    #print (frame.shape)
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    #print (frame.shape[:2])
    
    if ('nine_west' in cls):
        mask = return_rgb_mask(image_path, bx, cls)
    else:    
        if ('black' in cls):
        
            if('plain' in cls):
                
                lower_black1 = np.array([0,0,0])
                upper_black1 = np.array([180,255,35])
                
                lower_black2 = np.array([0,0,176])
                upper_black2 = np.array([180,255,255])
                
                mask1 = cv2.inRange(hsv, lower_black1, upper_black1)
                mask2 = cv2.inRange(hsv, lower_black2, upper_black2)
                
                mask = mask1 | mask2
            
            elif('ameligalanti' in cls):
                
                lower_black1 = np.array([20, 26, 24])
                upper_black1 = np.array([43, 54, 124])
                
                lower_black2 = np.array([80, 0, 96])
                upper_black2 = np.array([112, 62, 255])
                
                lower_black3 = np.array([95, 51, 0])
                upper_black3 = np.array([124, 112, 61])
                
                lower_black4 = np.array([9, 17, 0])
                upper_black4 = np.array([70, 166, 50]) 
                
                mask1 = cv2.inRange(hsv, lower_black1, upper_black1) 
                mask2 = cv2.inRange(hsv, lower_black2, upper_black2)
                mask3 = cv2.inRange(hsv, lower_black3, upper_black3)
                mask4 = cv2.inRange(hsv, lower_black4, upper_black4)
                
                mask = mask1 | mask2 | mask3 | mask4
                
            elif('backpack' in cls):
                
                lower_black = np.array([10, 8, 0])
                upper_black = np.array([120, 116, 225])
                mask = cv2.inRange(hsv, lower_black, upper_black)
            
        elif ('brown' in cls):
            
            if ('wrislet' in cls):
                
                lower_brown1 = np.array([95, 0, 0])
                upper_brown1 = np.array([175, 45, 90])
                
                lower_brown2 = np.array([35, 0, 85])
                upper_brown2 = np.array([65, 35, 190])
                
                lower_brown3 = np.array([40, 0, 120])
                upper_brown3 = np.array([100, 25, 255])
                
                lower_brown4 = np.array([16, 69, 8])
                upper_brown4 = np.array([43, 145, 114])
                
                lower_brown5 = np.array([4, 42, 86])
                upper_brown5 = np.array([25, 97, 255])
                
                mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
                mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
                mask3 = cv2.inRange(hsv, lower_brown3, upper_brown3)
                mask4 = cv2.inRange(hsv, lower_brown4, upper_brown4)
                mask5 = cv2.inRange(hsv, lower_brown5, upper_brown5)
                mask = mask1 | mask2 | mask3 | mask4 | mask5
            
            elif('messenger' in cls):
     
                lower_brown1 = np.array([2,20,50])
                upper_brown1 = np.array([17,220,255])
            
                lower_brown2 = np.array([160,20,50])
                upper_brown2 = np.array([180,220,255])
            
                lower_brown3 = np.array([1, 231, 120])
                upper_brown3 = np.array([21, 251, 200])
                
                mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
                mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
                mask3 = cv2.inRange(hsv, lower_brown3, upper_brown3)
                mask = mask1 | mask2 | mask3
                
        elif ('grey' in cls):
        
            lower_grey1 = np.array([0,0,69])
            upper_grey1 = np.array([29,76,255])
            
            lower_grey2 = np.array([46, 2, 81])
            upper_grey2 = np.array([105, 33, 255])
            
            lower_grey3 = np.array([102, 0, 61])
            upper_grey3 = np.array([160, 49, 236])
            
            mask1 = cv2.inRange(hsv, lower_grey1, upper_grey1)
            mask2 = cv2.inRange(hsv, lower_grey2, upper_grey2)
            mask3 = cv2.inRange(hsv, lower_grey3, upper_grey3)
            
            mask = mask1 | mask2 | mask3
        
        elif ('peach' in cls):
        
            lower_peach = np.array([0,10,100])
            upper_peach = np.array([25,255,255])
        
            mask = cv2.inRange(hsv, lower_peach, upper_peach)
        
        elif ('white' in cls):
        
            lower_white = np.array([0,0,200])
            upper_white = np.array([180,255,255])
            
            mask = cv2.inRange(hsv, lower_white, upper_white)
        
        elif ('red' in cls):
        
            lower_red1 = np.array([0,70,50])
            upper_red1 = np.array([10,255,255])
        
            lower_red2 = np.array([170,70,50])
            upper_red2 = np.array([180,255,255])
            
            lower_red3 = np.array([0, 0, 153])
            upper_red3 = np.array([177, 158, 255])
             
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
            mask = mask1 | mask2 | mask3
        
        elif ('blush' in cls):
        
            lower_blush1 = np.array([3, 11, 114])
            upper_blush1 = np.array([43, 123, 255])
            
            lower_blush2 = np.array([0, 0, 139])
            upper_blush2 = np.array([46, 46, 255])
            
            mask1 = cv2.inRange(hsv, lower_blush1, upper_blush1)
            mask2 = cv2.inRange(hsv, lower_blush2, upper_blush2)
        
            mask = mask1 | mask2
        
        mask = cv2.copyMakeBorder(mask, bx[0], height-bx[2], bx[1], width-bx[3], cv2.BORDER_CONSTANT,value=0)
    
    '''
    #print ("MASK BEFORE: ", mask.shape)
    #print ("BBOX: ", (bx[1],width-bx[3]), (bx[0],height-bx[2]))
    #print ("h:", height, "w:", width)
    #print ("MASK AFTER: ", mask.shape)
    
    cv2.imshow("mask", mask)
    cv2.imshow("image", cv2.imread(image_path))
    print(image_path)
    cv2.waitKey()
    '''
     
    return mask
    
def get_hsv_masks(image_path):
    
    split = image_path.split('JPEGImages')
    
    annot_path = split[0]+'Annotations'+split[1][:-3]+'xml'
    
    #print (image_path, annot_path)
    
    tree = ET.parse(annot_path)
    root = tree.getroot()         
    
    num_masks, width, height = len(root.findall('object')), int(root.find('size').find('width').text), int(root.find('size').find('height').text)
    
    #print (height, width)
    
    image_masks, classes = np.zeros((height, width, num_masks), dtype=np.uint8), []
    
    for i, obj in enumerate(root.findall('object')):
    
        cls = obj.find('name').text
        bx = [int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymax').text), int(obj.find('bndbox').find('xmax').text)]
        image_masks[:,:,i] = return_hsv_mask(image_path, np.array(bx), cls)
        classes.append(cls)
    
    image_masks = np.array(image_masks/255, dtype=np.uint8)
        
    '''
    orig_im = Image.open(image_path)
    for i in range(image_masks.shape[-1]):
        mask = Image.fromarray(np.dstack((image_masks[...,i], image_masks[...,i], image_masks[...,i])), 'RGB')
        Image.blend(orig_im, mask, 0.75).show()
    '''
    
    return image_masks, classes

def test_time():
    
    tic = time.time()
    
    for f in glob.glob(os.getcwd()+'/Data/handbag_images/JPEGImages/*'):
        masks, classes = get_hsv_masks(f)
        cv2.waitKey()
        
    #seg, bb = find_object_bbox_masks(os.getcwd()+'/Data/bags/black_ameligalanti/2017-L7-CK2-20780452-01-1.jpg')

    #Image.fromarray(return_mask('/home/hans/Desktop/Vision Internship/github/Mask_RCNN/Data/handbag_images/JPEGImages/bot3.png', np.array([1, 1, 178, 192]), 'black_ameligalanti'), 'L').show()
    cv2.destroyAllWindows()
    print (time.time()-tic)

#test_time()
