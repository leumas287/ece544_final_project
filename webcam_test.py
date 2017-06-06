# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:20:23 2017

@author: Dakota
"""

import numpy as np

import cv2





# In[ ]:



cap = cv2.VideoCapture(0)





# In[ ]:



while(True):

    # Capture frame-by-frame

    ret, frame = cap.read()



    # Our operations on the frame come here

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # Display the resulting frame

    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break





# In[ ]:



# When everything done, release the capture

cap.release()

cv2.destroyAllWindows()