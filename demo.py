"""
Example script to use NovoOpenDetector detection module for spot picking operation

Author: Dimitrios Arapis (DTAI)
Date: 2023-05-31
"""

import os
import cv2 
#from NovoOpenDetector.detection import NovoOpenDetector
from detection import NovoOpenDetector

#BUILD MODEL AND DETECT
MODEL = 'full' #Options: ['full', 'simple']. 
CLASSES = ['bottle', 'banana'] # classes to detect
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
image = cv2.imread("data/examples/example.jpg")  #image from camera for spot

spot_detection = NovoOpenDetector(model='full', classes = CLASSES, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD)

for i in range(0,10):
    result_image, object, confidence, centroid =  spot_detection.detect(image)
    if object is not None:
        print(40*'*')
        print(f"Found a {object.upper()} with confidence {confidence} % at location (x={centroid[0]}, y={centroid[1]})")
        print(40*'*')
    else: 
        print(40*'*')
        print("Could not find any objects")
        print(40*'*')

#cv2.imshow("Result", image)
#cv2.waitKey(0)