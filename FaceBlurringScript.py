import cv2
import numpy as np 
from random import randint

filenum=randint(100,450)
imname='image_0'+str(filenum)+'.jpg'
colorim=cv2.imread(imname,1)
raw_image=cv2.imread(imname,1) #loads a random grayscale image from directory
greyscale=cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
equalized=cv2.equalizeHist(greyscale)
#Haar-cascade classifier for face, pre-built in OpenCV
face_cascade=cv2.CascadeClassifier('C:\Users\harse\Desktop\Python Projects\opencv-3.4.1\data\haarcascades\haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(equalized,1.05,5,5) #outputs a list of rectangles of face bounding boxes
for (x,y,w,h) in faces: #faces is a list containing bounding box coordinates and dimensions
    cv2.rectangle(raw_image,(x,y),(x+w,y+h),(255,0,0),2)

for boxnum in range(0,len(faces),1):
    for j in range(faces[boxnum,0],faces[boxnum,0]+faces[boxnum,2],1):#interating through the bounding box 
        for i in range(faces[boxnum,1], faces[boxnum,1]+faces[boxnum,3],1):
            raw_image[i,j]=raw_image[i,j]/2

cv2.imshow('Burred',raw_image)
cv2.imshow('Original',colorim)
cv2.waitKey(0)
cv2.destroyAllWindows()