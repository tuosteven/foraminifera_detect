# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:42:54 2021

@author: acer
"""


import cv2
#import ImageProcess
import numpy as np 

#截圖長寬 
w=150
h=150


location = "E:/training_data"
start = 1 # start pic
end = 700   # end pic
start_index = 0



def ImageDetection(location, start, end, index):
    new_index = index
    
    for i in range(start, end+1):
        print(i)
        address = location + "/pic_" + str(i) + ".jpg"
        try:
            img = cv2.imread(address)
            new_index = Detection(img, new_index)
        except :
            continue   
        
        #print(new_index)
    
    return new_index

def Detection(ori_img, index):

    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 10)
    #cv2.imshow('img1', img)

    imgmax=img.max()

    #print(imgmax)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    threshild_low=(imgmax*1.304)-145
    #print(threshild_low)


    ret, th1 = cv2.threshold(img,threshild_low,255,cv2.THRESH_BINARY)
    #cv2.imshow('img', th1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #kernal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #執行影象形態學
    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('erode dilate', closed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    # 腐蝕 4 次，去掉細節
    closed = cv2.erode(closed, None, iterations=5)
    #cv2.imshow('erode dilate', closed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # 膨脹 14 次，讓輪廓突出
    closed = cv2.dilate(closed, None, iterations=2 )
    #cv2.imshow('erode dilate', closed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    (_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    possible_img = ori_img.copy()
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect1 = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    Box = np.int0(cv2.boxPoints(rect))
    Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    area = cv2.contourArea(Box)
    #cv2.imshow('Final_img', Final_img)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()



    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
    #  
        rect = cv2.minAreaRect(c)

        rect = (rect[0], (w,h), 0)
        Box = np.int0(cv2.boxPoints(rect))
      
        
      
        #area1 = cv2.contourArea(Box)
        #possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        
        #print(Box)
        #print('\n')
        x=Box[1][0]
        y=Box[1][1]
        if (x<0 or x>(1280-w) or y<0 or y>(720-h)):continue
    
        crop_img=ori_img[y:y+h,x:x+w]
        
        filename = "output/%d.jpg"%(index)
        
        print(filename)
        cv2.imwrite(filename,crop_img)
        #cv2.imshow("crop_img",crop_img)
        index += 1
            
     
    
    #cv2.imshow('possible_img', possible_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return index


ImageDetection(location, start, end, start_index)