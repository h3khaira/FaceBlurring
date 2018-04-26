import cv2
import numpy as np 
from random import randint

imname='Oscar.jpg'

raw_image=cv2.imread(imname,1) #loads a random grayscale image from directory
#greyscale=cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
#equalized=cv2.equalizeHist(greyscale)
#Haar-cascade classifier for face, pre-built in OpenCV
face_cascade=cv2.CascadeClassifier('C:\Users\harse\Desktop\Python Projects\opencv-3.4.1\data\haarcascades\haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(equalized,1.05,5,5) #outputs a list of rectangles of face bounding boxes
i=1
crop_image=[[]]
boxnum=0
blurringIndex=5
for (x,y,w,h) in faces: #faces is a list containing bounding box coordinates and dimensions
    for col in range(faces[boxnum,0],faces[boxnum,0]+faces[boxnum,2],blurringIndex): #iterates through the bounding box 
        for row in range(faces[boxnum,1], faces[boxnum,1]+faces[boxnum,3],blurringIndex): 
            for pixelcol in range(col,col+blurringIndex,1): #iterates through "big" pixel clusters
                for pixelrow in range(row,row+blurringIndex,1):
                    raw_image[pixelrow,pixelcol]=raw_image[row,col] #set the value of pixels in that cluster to one colour
    boxnum=boxnum+1   
    i=i+1
    #cv2.rectangle(raw_image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Face Detected',raw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()