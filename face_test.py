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
import dlib
import argparse


detector = dlib.get_frontal_face_detector()
mymodel = load_model('mymodel.h5')
cap=cv2.VideoCapture(2)

while cap.isOpened():
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) > 0:
        print("Faces:",len(faces))
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.imwrite('temp.jpg',frame[y1:y2,x1:x2])
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mymodel.predict_classes(test_image)[0][0]
            prob=mymodel.predict_proba(test_image)
            if pred==1:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.imwrite('NOMASK_face.jpg',frame[y1:y2,x1:x2])
            else:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    else:
        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue
    datet=str(datetime.datetime.now())
    cv2.putText(frame,datet,(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    final_img = cv2.resize(frame, (960,540))
    cv2.imshow('img',final_img)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
