# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 00:20:06 2017

@author: Dakota
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:02:57 2017

@author: Dakota
"""# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:19:28 2017

@author: Dakota
"""
import numpy as np
import cv2

'''
used to grab actual camera frame dimensions
#fWidth = cap.get(3)
#fHeight = cap.get(4)
'''
def update_shape(contours):
    approx = cv2.approxPolyDP(contours,0.01*cv2.arcLength(contours,True),True) 
    if len(approx) == 4:
        return "square"
            
    elif len(approx) > 8 and len(approx) < 12:
        return "cylinder"        
    else:
        return "invalid"
   
def update_box_points(contours):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="int")
    return box
    
    
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
 
def get_dimensions(pts,known_width):
    dimensions = {"height_px":None, "width_px":None, "w_In":None}
    
    # obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the bounding box, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    dimensions["width_px"] = maxWidth
    dimensions["w_In"] = maxWidth/known_width
              
    # compute the height of the bounding box, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dimensions["height_px"] = maxHeight
    return dimensions

       
def find_frame_pos(centerX):
    position = ''
    
    if (int(centerX) > 0) and (int(centerX) < 213):
        position = "left"  
    
    elif (int(centerX) > 213) and (int(centerX) < 426):
        position = "middle"
     
    else:    
        position ="right"   
    
    return position

def distance_to_camera(knownWidth, focalLength, percieved_width):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / percieved_width

def update_distance(known_width,focalLength, perWidth):
    inches = distance_to_camera(known_width, focalLength, perWidth)
    return inches

#Camera Calibration
#Known width/height of coke can
KNOWN_CAN_WIDTH          = 2.598
KNOWN_TARGET_WIDTH       = 13.937
KNOWN_CAN_PIXEL_WIDTH    = 133
KNOWN_TARGET_PIXEL_WIDTH = 344
KNOWN_CAN_DISTANCE       = 12 #empirically derived distance from lens to coke can
KNOWN_TARGET_DISTANCE    = 26
focalLength_can = (KNOWN_CAN_PIXEL_WIDTH * KNOWN_CAN_DISTANCE) / KNOWN_CAN_WIDTH
focalLength_target = (KNOWN_TARGET_PIXEL_WIDTH * KNOWN_TARGET_DISTANCE) / KNOWN_TARGET_WIDTH

cap = cv2.VideoCapture(0)
shape =' '
results ={}
position=' '
distance = 0
cnt_length = 0

# define range of red color in HSV
lower_red = np.array([17, 15, 100], dtype=np.uint8)
upper_red = np.array([50, 56, 200], dtype=np.uint8)
# define range of green color in HSV
lower_green = np.array([34, 50, 50], dtype=np.uint8)
upper_green = np.array([80, 220, 200], dtype=np.uint8)
# define range of blue color in HSV
lower_blue = np.array([92, 0, 0], dtype=np.uint8)
upper_blue = np.array([124, 255, 255], dtype=np.uint8)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only red colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue  = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_zero = np.zeros(hsv.shape[:2],np.uint8)
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    # RGB to Gray scale conversion
    img_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
    # Thresholding the image
    ret,thresh_image = cv2.threshold(noise_removal,0,255,cv2.THRESH_OTSU)
    
    # Applying Canny Edge detection
    canny_image = cv2.Canny(thresh_image,250,255)
    # Display Image
    canny_image = cv2.convertScaleAbs(canny_image)

    # dilation to strengthen the edges
    kernel = np.ones((3,3), np.uint8)
    # Creating the kernel for dilation
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
   
    
    #find contour data
    __,contours,__ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    pt = (180, 3 * frame.shape[0] // 4)
    for cnt in contours:
        	# if the contour is not sufficiently large, ignore it
            if cv2.contourArea(cnt) < 100:
                continue
            
            shape = update_shape(cnt)
            if (shape == "square"):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype="int")
                cX = np.average(box[:, 0])
                # compute the center of the bounding box
                cX = np.average(box[:, 0])
                results = get_dimensions(box, KNOWN_TARGET_WIDTH)
                position = find_frame_pos(cX)
                distance = update_distance(KNOWN_TARGET_WIDTH, focalLength_can, results["width_px"]) 
                print(results)
                print(position)
                print("dist from camera:", distance)
                if cv2.countNonZero(mask_blue):
                    print("Blue detected")
                    mask_green = mask_zero
                    mask_red = mask_zero
                    cv2.drawContours(frame,[box],0,(0,0,255),2)
                else:
                    continue
                
            elif (shape == "cylinder"):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype="int")
                cX = np.average(box[:, 0])
                # compute the center of the bounding box
                cX = np.average(box[:, 0])
                results = get_dimensions(box, KNOWN_CAN_WIDTH)
                position = find_frame_pos(cX)
                distance = update_distance(KNOWN_CAN_WIDTH, focalLength_can, results["width_px"]) 
                print(results)
                print(position)
                print("dist from camera:", distance)
                if cv2.countNonZero(mask_red):
                    print("Red detected")
                    mask_green = mask_zero
                    mask_blue = mask_zero
                    cv2.drawContours(frame,[box],0,(0,0,255),2)
                else:
                    continue
            
          
    cv2.imshow("Soda Can", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

