import skimage.io as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os


hs_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/HS.xml'))

image = cv2.imread('/home/lenovo/Documents/abc1.jpeg')

mar1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
imageio.imshow(mar1)

hs=hs_cascade.detectMultiScale(mar1, 1.3, 5)

for (x,y,w,h) in hs:
    cv2.rectangle(image, (x,y), (x+w, y+h), (12,150,100),2)

gozler=[]
for (x,y,w,h) in hs:
    gozler.append(mar1[y:y+h, x:x+w])
imageio.imshow(gozler[0])

for gz in gozler:
   plt.imshow(gz)
   plt.show()  
   pd.DataFrame({'HS':str(gozler[0])},index=[0]).to_csv('hsdata.csv')
