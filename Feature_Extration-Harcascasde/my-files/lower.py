import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/lenovo/Documents/image1.jpeg',cv2.IMREAD_COLOR)

low_cascade = cv2.CascadeClassifier('/home/lenovo/Documents/haarcascades/haarcascade_lowerbody.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

low = low_cascade.detectMultiScale(gray, 1.1 , 3)
   

for (x,y,w,h) in low:
    cv2.rectangle(img, (x,y), (x+w, y+h), (12,150,100),2)
    
cv2.imshow('image',img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
