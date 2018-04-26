import cv2
import numpy as np 
import sys, math
from random import randint

def expand(face):
    rows, columns, channels = face.shape
    for row in range(0,rows):
        for col in range(0,columns):
            face[row,col] = face[0,0]

def pixelface(face, bluramount):
    rows, columns, channels = face.shape
    #fill
    if (rows <= bluramount or columns <= bluramount):
        expand(face)
        return
    
    #divide into 4 pieces
    facepart = []
    for j in range(0,2):
        for i in range(0,2):
            #coordinates
            xstart  = int(math.floor(i * (rows/2)))
            xend    = int(math.ceil((i+1)*(rows/2)))
            ystart  = int(math.floor(j*(columns/2)))
            yend    = int(math.ceil((j+1)*(columns/2)))

            facepart.append(face[xstart:xend, ystart:yend])
            # print(len(facepart))
            # print("part " + str(i*(rows/2)) + " to " + str((i+1)*(rows/2)))
            # print("part " + str(j*(columns/2)) + " to " + str((j+1)*(columns/2)) + "\n")

    #recursive pieces
    for i in range(0,4):
        pixelface(facepart[i], bluramount)


imname=sys.argv[1]
colorim=cv2.imread(imname,1)
raw_image=cv2.imread(imname,1) #loads a random grayscale image from directory
greyscale=cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
equalized=cv2.equalizeHist(greyscale)

#Haar-cascade classifier for face, pre-built in OpenCV
face_cascade=cv2.CascadeClassifier('D:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
#face_cascade=cv2.CascadeClassifier('C:\Users\harse\Desktop\Python Projects\opencv-3.4.1\data\haarcascades\haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(equalized,1.05,5,5) #outputs a list of rectangles of face bounding boxes
i=1
crop_image=[[]]
for (x,y,w,h) in faces: #faces is a list containing bounding box coordinates and dimensions
    crop_image.append(raw_image[y:y+h,x:x+w])

    #blur face
    pixelface(crop_image[i], int(0.08*(crop_image[i].shape[0])))

    i=i+1
    cv2.rectangle(raw_image,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow('Face Detected', raw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

