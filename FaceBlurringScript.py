import cv2
import numpy as np 
from random import randint

filenum=randint(100,450)
imname='image_0'+str(filenum)+'.jpg'
raw_image=cv2.imread(imname,1) #loads a random grayscale image from directory
greyscale=cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
#Haar-cascade classifier for face, pre-built in OpenCV
face_cascade=cv2.CascadeClassifier('C:\Users\harse\Desktop\Python Projects\opencv-3.4.1\data\haarcascades\haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(greyscale,1.05,5,5) #outputs a list of rectangles of face bounding boxes
for (x,y,w,h) in faces: #faces is a list containing bounding box coordinates and dimensions
    cv2.rectangle(raw_image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Color',raw_image)
faceSplice=[]
for i in range(faces[0,0],faces[0,0]+faces[0,2],1):#interating through the bounding box 
    for j in range(faces[0,1], faces[0,1]+faces[0,3],1):
        raw_image[i,j]=(255,255,255)
cv2.imshow('Color',raw_image)

cv2.waitKey(0)
cv2.destroyAllWindows()