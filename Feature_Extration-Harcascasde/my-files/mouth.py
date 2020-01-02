import skimage.io as imageio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


Mouth_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/Mouth.xml'))

mar=cv2.imread("/home/lenovo/Documents/abc.jpeg")
mar1=cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
imageio.imshow(mar1)

gray=cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
imageio.imshow(gray)
    
mouth=Mouth_cascade.detectMultiScale(mar1, 1.3, 5)

print(mar.dtype)
print(mar.shape)

gozler=[]
for (x,y,w,h) in mouth:
    gozler.append(mar1[y:y+h, x:x+w])
imageio.imshow(gozler[0])
imageio.imshow(gozler[1])

for gz in gozler:
   plt.show()  
   pd.DataFrame({'mouth':str(gozler[0])},index=[0,1]).to_csv('mouthdata.csv')


