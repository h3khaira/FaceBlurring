def fkface(face):
    print("method")
    cv2.imshow('inside', face)
    rows, columns, channels = face.shape

    #go thru pixels
    for row in range(0,(rows/2)-1):
        for col in range(0,columns-1):
            temp = face[row, col]
            face[row, col] = face[(rows-row)-1, col]
            face[(rows-row)-1, col] = temp
            col += 1
    cv2.imshow('switch', face)
    
import cv2
import numpy as np 
import sys
from random import randint

filenum=randint(100,450)
#imname='image_0'+str(filenum)+'.jpg'
imname=sys.argv[1]
colorim=cv2.imread(imname,1)
raw_image=cv2.imread(imname,1) #loads a random grayscale image from directory
greyscale=cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
equalized=cv2.equalizeHist(greyscale)
#Haar-cascade classifier for face, pre-built in OpenCV
face_cascade=cv2.CascadeClassifier('D:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(equalized,1.05,5,5) #outputs a list of rectangles of face bounding boxes
i=1
crop_image=[[]]
for (x,y,w,h) in faces: #faces is a list containing bounding box coordinates and dimensions
    crop_image.append(raw_image[y:y+h,x:x+w])
    #cv2.imshow('Cropped '+str(i),crop_image[i])
    fkface(crop_image[i])
    i=i+1
    cv2.rectangle(raw_image,(x,y),(x+w,y+h),(255,0,0),2)
  
#for boxnum in range(0,len(faces),1):
    #for j in range(faces[boxnum,0],faces[boxnum,0]+faces[boxnum,2],1): #iterating through the bounding box 
        #for i in range(faces[boxnum,1], faces[boxnum,1]+faces[boxnum,3],1):


cv2.imshow('Face Detected', raw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

