import numpy as np
import cv2
from imutils import paths
import os

f = open("face_data.csv", "w+")
f.write("x,y,w,h,url \n")
imagePaths = paths.list_images('images')
face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascades/haarcascade_frontalface_default.xml'))

for imgpath in imagePaths:
    path = os.path.abspath(imgpath)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        if (x,y,w,h):
            face_data = "{},{},{},{},{} \n".format(x,y,w,h, path)
            f.write(face_data)
            print(face_data)
f.close()
