import cv2
import numpy as np

image_hsv = None   
pixel = (20,60,80) 

max_hsv, min_hsv = np.zeros(3), np.ones(3)*255

# mouse callback function
def pick_color(event,x,y,flags,param):
    
    global max_hsv, min_hsv
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        
        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        
        for i in range(3):
            if(max_hsv[i]<upper[i]):
                max_hsv[i]=upper[i]
            if(min_hsv[i]>lower[i]):
                min_hsv[i]=lower[i]

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)
        print (max_hsv, min_hsv)

def main():
    import sys
    global image_hsv, pixel # mouse callback

    image_src = cv2.imread(sys.argv[1])  
    
    if image_src is None:
        print ("File Read Error")
        return
    
    cv2.imshow("bgr",image_src)

    ## NEW ##
    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    # now click into the hsv img , and look at values:
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
