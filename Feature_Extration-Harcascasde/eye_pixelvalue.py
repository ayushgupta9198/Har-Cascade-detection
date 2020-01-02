import skimage.io as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


eye_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_eye.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/Nose.xml'))
face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_frontalface_default.xml'))
frontal_face = cv2.CascadeClassifier("/home/lenovo/Documents/haarcascades/haarcascade_frontalface_alt2.xml")
Mouth_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/Mouth.xml'))

image=cv2.imread("/home/lenovo/Documents/abc.jpeg")
mar1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
imageio.imshow(mar1)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
imageio.imshow(mar1)

eye=eye_cascade.detectMultiScale(mar1, 1.3, 5)
nose=nose_cascade.detectMultiScale(mar1, 1.3, 5)

gozler=[]
for (x,y,w,h) in eye,nose
    gozler.append(mar1[y:y+h, x:x+w])
imageio.imshow(gozler[0])

for gz in gozler:
   plt.imshow(gz)
   pd.DataFrame({'leftEye':str(gozler[0]), 'rightEye':str(gozler[1])},index=[0,1]).to_csv('eyesdata.csv')
   pd.DataFrame({'nose':str(gozler[0])},index=[0]).to_csv('nosedata.csv')


