# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:05:00 2017

@author: Dakota
"""

import cv2
#import sys
import numpy as np
import imutils

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.06, minNeighbors=3, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


#cascPath = sys.argv[1]
cascade = cv2.CascadeClassifier('can_lbp.xml')

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    rects = detect(gray, cascade)
    vis = frame.copy()
    draw_rects(vis, rects, (0, 255, 0))
    
    '''
    for x1, y1, x2, y2 in rects:
        roi = gray[y1:y2, x1:x2]
        vis_roi = vis[y1:y2, x1:x2]
        subrects = detect(roi.copy(), nested)
        draw_rects(vis_roi, subrects, (255, 0, 0))
    '''

    # Display the resulting frame
    cv2.imshow('Video', vis)
    #cv2.imshow('Video', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
