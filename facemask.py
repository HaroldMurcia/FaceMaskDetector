# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 11:50:00 2020

@author: www.haroldmurcia.com
"""


import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime


# IMPLEMENTING LIVE DETECTION OF FACE MASK

cap=cv2.VideoCapture(2)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mymodel = load_model('mymodel.h5')
pred_1=0
pred_2=0
pred_3=0
while cap.isOpened():
    _,img=cap.read()
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image=image.load_img('temp.jpg',target_size=(150,150,3))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        pred=mymodel.predict_classes(test_image)[0][0]
        prob=mymodel.predict_proba(test_image)
        if pred==1 and pred_1==1 and pred_2==1 and pred_3==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'SIN MASCARA',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        elif pred==0 and pred_1==0 and pred_2==0 and pred_3==0:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'CON MASCARA',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        pred_3=pred_2
        pred_2=pred_1
        pred_1=pred
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow('img',img)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
