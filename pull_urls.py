# -*- coding: utf-8 -*-
#!

import urllib.request
import cv2
import numpy as np
import os

def pull_samples(url_link, filename, url_num, height, width):
    images_link = url_link   #replace the wnid="<insert synset #>"
    image_urls = urllib.request.urlopen(images_link).read().decode()
    pic_num = url_num  # hardcoded placeholder, for appending further images
    
    for i in image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, filename+str(pic_num)+".jpg")
            img = cv2.imread(filename+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE) #relative path to destination folder
            # negative samples should be larger than postive sample images
            '''
            #to preserve aspect ratio (use if needed)
            r = 100.0 / img.shape[1]
            dim = (height, int(img.shape[0] * r))
            resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            '''
            resized_image = cv2.resize(img, (height, width))
            cv2.imwrite(filename+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))
            
            
def find_errors(filename):
    for file_type in [filename]:
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



pull_samples("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04255586", "resized_positives/", 1, 75, 75)
find_errors("resized_positives/")