# -*- coding: utf-8 -*-
#!

import urllib.request
import cv2
import numpy as np
import os

def pull_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03593526'   #replace the wnid="<insert synset #>"
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1314  # hardcoded placeholder, for appending further images
    
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "resized_negatives/"+str(pic_num)+".jpg")
            img = cv2.imread("resized_negatives/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE) #relative path to destination folder
            # should be larger than postive sample images
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("resized_negatives/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e)) 

def find_errors():
    for file_type in ['resized_negatives']:
        for img in os.listdir(file_type):
            for badPic in os.listdir('rejects'): #'rejects' is src folder of error jpgs copy and pasted from dst folder of pulled images
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    badPic = cv2.imread('rejects/'+str(badPic))
                    question = cv2.imread(current_image_path)
                    if badPic.shape == question.shape and not(np.bitwise_xor(badPic,question).any()):
                        print('Deleting jpg error')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))


#pull_images()
find_errors()