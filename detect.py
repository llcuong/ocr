import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

def Segment():
    imgdir = cv2.imread(f'Temp/scan.jpg',0)
    model = load_model('weights/line.h5')
    img= imgdir
    img= imgdir
    img = img[int(img.shape[0]*0.4):img.shape[0], int(0.29*img.shape[1]):img.shape[1]]
    ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    img=cv2.resize(img,(512,512))
    img= np.expand_dims(img,axis=-1)
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)
    plt.imsave('Temp/test_img_mask.JPG',pred)
    coordinates=[]
    img = cv2.imread('Temp/test_img_mask.JPG',0) 
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    ori_img= imgdir
    ori_img = ori_img[int(ori_img.shape[0]*0.4):ori_img.shape[0], int(0.29*ori_img.shape[1]):ori_img.shape[1]]
    ori_img=cv2.resize(ori_img,(512,512))
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(ori_img, (x, y), (x+w,y+h), 255, 1)
        coordinates.append([x,y,(x+w),(y+h)])
    images = []
    ori = imgdir
    ori = ori[int(ori.shape[0]*0.4):ori.shape[0], int(0.29*ori.shape[1]):ori.shape[1]]
    imgs =cv2.resize(ori,(512,512))
    for i in range(0, len(coordinates)):
        if -coordinates[i][0]+coordinates[i][2] > 40 and abs(-coordinates[i][1]+coordinates[i][3]) > 30:
            imgx = imgs[coordinates[i][1]:coordinates[i][3]+1, coordinates[i][0]:coordinates[i][2]+1]
            images.append(imgx)
            cv2.imwrite('Trash/'+str(i)+'.jpg',imgx)
    output = list(reversed(images))
    return output