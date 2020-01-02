import skimage.io as imageio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


nose_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/Nose.xml'))
face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_frontalface_default.xml'))

mar=cv2.imread("/home/lenovo/Documents/abc1.jpeg")
mar1=cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
imageio.imshow(mar1)

gray=cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
imageio.imshow(gray)
    
nose=nose_cascade.detectMultiScale(mar1, 1.3, 5)

print(mar.dtype)
print(mar.shape)

gozler=[]
for (x,y,w,h) in nose:
    gozler.append(mar1[y:y+h, x:x+w])
imageio.imshow(gozler[0])

for gz in gozler:
   plt.imshow(gz)
   plt.show()  
   pd.DataFrame({'nose':str(gozler[0])},index=[0]).to_csv('nosedata.csv')


