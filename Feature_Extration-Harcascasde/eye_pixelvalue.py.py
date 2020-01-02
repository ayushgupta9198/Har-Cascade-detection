import skimage.io as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


lower_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_lowerbody.xml'))
#face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_frontalface_default.xml'))

mar=cv2.imread("/home/lenovo/Documents/abc1.jpeg")
mar1=cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
imageio.imshow(mar1)

gray=cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
imageio.imshow(gray)

eye=eye_cascade.detectMultiScale(mar1, 1.3, 5)

gozler=[]
for (x,y,w,h) in eye:
    gozler.append(mar1[y:y+h, x:x+w])
imageio.imshow(gozler[0])
imageio.imshow(gozler[1])

for gz in gozler:
   plt.imshow(gz)
   plt.show()  
   pd.DataFrame({'leftEye':str(gozler[0]), 'rightEye':str(gozler[1])},index=[0,1]).to_csv('eyesdata.csv')


