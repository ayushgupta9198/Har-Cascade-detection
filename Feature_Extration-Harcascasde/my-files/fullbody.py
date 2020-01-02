from matplotlib import pyplot as plt
import skimage.io as imageio
import pandas as pd
import numpy as np
import cv2

fullbody = cv2.CascadeClassifier("/home/lenovo/Documents/haarcascades/haarcascade_fullbody.xml")

image = cv2.imread("i.jpeg")

gray=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
imageio.imshow(gray)

forehead=fullbody.detectMultiScale(gray, 1.2, 3)
print(forehead)

gozler=[]
for (x,y,w,h) in forehead:
    gozler.append(gray[y:y+h, x:x+w])
imageio.imshow(gozler[0])
# imageio.imshow(gozler[1])

for gz in gozler:
   plt.imshow(gz)
   plt.show()  
   pd.DataFrame({'fullbody':str(gozler[0]), 'forehead':str(gozler[1])},index=[0,1]).to_csv('fullbodydata.csv')