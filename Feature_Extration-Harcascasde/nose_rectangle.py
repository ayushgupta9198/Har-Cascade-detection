import skimage.io as imageio
import numpy as np
import pandas as pd
import dlib
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import os
import csv

nose_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/Nose.xml'))
face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_frontalface_default.xml'))


# Read the image
image = cv2.imread("/home/lenovo/Documents/abc1.jpeg")
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
# Draw a rectangle around the faces
face = face_cascade.detectMultiScale(a, 1.3, 5)

for (x,y,w,h) in face:
  cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
  roi_gray = a[y:y+h, x:x+w]
  roi_color = image[y:y+h, x:x+w]
  nose = nose_cascade.detectMultiScale(roi_gray)
  for (ex,ey,ew,eh) in nose:
  	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow("Nose found" ,image)
print(image.dtype)
print(image.shape)

# gozler=[]
# for (x,y,w,h) in nose:
#     gozler.append(a[y:y+h, x:x+w])
# imageio.imshow(gozler[0])
# #imageio.imshow(gozler[1])

# for gz in gozler:
#    plt.imshow(gz)
#    plt.show()  

