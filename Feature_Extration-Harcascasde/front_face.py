from matplotlib import pyplot as plt
import skimage.io as imageio
import pandas as pd
import numpy as np
import cv2

frontal_face = cv2.CascadeClassifier("/home/lenovo/Documents/haarcascades/haarcascade_frontalface_alt2.xml")

image = cv2.imread("abc.jpeg")
mar1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face = frontal_face.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=(cv2.CASCADE_SCALE_IMAGE))

gozler=[]
for (x, y, w, h) in face:
    cv2.rectangle(mar1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    gozler.append(mar1[y:y+h, x:x+w])
    imageio.imshow(gozler[0])

for gz in gozler:
    plt.imshow(gz)
    plt.show()  
    pd.DataFrame({'front_face':str(gozler[0])},index=[0]).to_csv('frontdata.csv')
