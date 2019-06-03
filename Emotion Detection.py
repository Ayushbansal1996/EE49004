
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import contextlib
import random
import sys
from sklearn import preprocessing
from skimage import feature
import time
import datetime as dt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from sklearn import datasets, svm, metrics
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import svm
eps =1e-7
points=256
radius=8
gamma=1.2


# In[2]:


def adjust_gamma(image):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# In[3]:


def dog(image):
    blur1= cv2.GaussianBlur(image,(5,5),0)
    blur2= cv2.GaussianBlur(image,(3,3),0)
    image=blur1-blur2
    return image


# In[4]:


def describe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    img=dog(adjust_gamma(equ))
    lbp = feature.local_binary_pattern(img,points,radius, method="default")
    lbp=lbp.reshape(-1)
    return lbp


# In[5]:


def data_set(data,label):

    for file in os.listdir("emotion"):
        if file.endswith(".png"):
            img = cv2.imread("emotion/" + file) 
            img = cv2.resize(img, (200, 200)) 
            if file.startswith("Smiling"):
                label.append("Smiling")
            if file.startswith("Angry"):
                label.append("Angry")
            if file.startswith("Laughing"):
                label.append("Laughing")
            if file.startswith("Normal"):
                label.append("Normal")
            lbp=describe(img)
            data.append(lbp)
    return data,label


# In[6]:


def Image_Capture(img_count,StrA_F,StrA):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Training")
    img_counter = 0
    frame_read=0
    img_counter = 0
    key_status=0            
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
                                            gray, 
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                            )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame_crop = frame[ y:y+h , x:x+w ]
        
        cv2.imshow("Training", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            key_status=1
        
        if key_status==1:
            if frame_read<30:
                frame_read=frame_read+1
            else:
                img_name = "C:/Users/ayus/"+StrA_F+"/"+StrA+"_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame_crop)
                print("{} written!".format(img_name))
                img_counter += 1
                frame_read=0
                if img_counter<img_count:
                    key_status=1
                else:
                    key_status=0
    cam.release()
    cv2.destroyAllWindows()


# In[7]:


def model_training(X_train,Y_train):
    model = LinearSVC(C=100.0, random_state=50, max_iter = 8000)
    model.fit(X_train, Y_train)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model


# In[8]:


def model_testing(X_test,Y_test,model):
    expected = Y_test
    predicted = model.predict(X_test)  
    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)
    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


# In[9]:


def Start_Video():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    temp = 0
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Emotion_Recognition")
    img_counter = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                            )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame_crop = frame[ y:y+h , x:x+w ]
        
        
        img = cv2.resize(frame_crop, (200, 200))
        cv2.imwrite('facelogs/face.jpg' ,img)
        image = cv2.imread("facelogs/face.jpg" ) 
        lbp=describe(img)
        lbp=lbp.reshape(1,-1)
        result = loaded_model.predict(lbp)
        if result == "Normal":
            cv2.putText(frame, "Normal", (x, y),  cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))  
        elif result == "Laughing":
            cv2.putText(frame, "Laughing", (x, y),  cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))  
        elif result == "Smiling":
            cv2.putText(frame, "Smiling", (x, y),  cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))  
        elif result == "Angry":
            cv2.putText(frame, "Angry", (x, y),  cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))  
        
        cv2.imshow("Emotion_Recognition", frame)
            
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            print("Escape hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()


# In[10]:


def Face_Capture():
    Image_Capture(20,"emotion","Laughing")
    Image_Capture(20,"emotion","Angry")
    Image_Capture(20,"emotion","Smiling")
    Image_Capture(20,"emotion","Normal")


# In[11]:


def Training():
    data=[]
    label=[]
    data,label=data_set(data,label)
    print(len(data))
    print(len(label))
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.15, random_state=42)
    model=model_training(X_train,Y_train)
    model_testing(X_test,Y_test,model)


# In[12]:


Face_Capture()


# In[13]:


Training()


# In[14]:


Start_Video()

