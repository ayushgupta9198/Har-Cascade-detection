import skimage.io as imageio
import numpy as np
import pandas as pd
import dlib
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import os
import csv

hs_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/HS.xml'))

image = cv2.imread("/home/lenovo/Documents/image2.jpeg")

a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	

lower_cascade = lower_body_cascade.detectMultiScale(a, 1.3, 5)

for (x,y,w,h) in lower_cascade:
  cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
  roi_gray = a[y:y+h, x:x+w]
  roi_color = image[y:y+h, x:x+w]

cv2.imshow("found" ,image)
# print(image.dtype)
# print(image.shape)

gozler=[]
for (x,y,w,h) in lower_cascade:
    gozler.append(a[y:y+h, x:x+w])
imageio.imshow(gozler[0])
#imageio.imshow(gozler[1])

for gz in gozler:
   plt.imshow(gz)
   plt.show() 
   pd.DataFrame({'HS':str(gozler[0])},index=[0]).to_csv('hsdata.csv')
 
