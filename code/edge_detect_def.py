# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:42:54 2021

@author: acer
"""
import time 
import predict
import cv2
#import ImageProcess
import numpy as np 
import os
import matplotlib.pyplot as plt
# In[]:
#截圖長寬 
w=150
h=150

#測試檔案所在之資料夾
#location = "/Users/juechenxun/Desktop/nsysu/ML/model/foraminiera_detect-main/uncrop"
#file=os.listdir(location)
start_index = 0
total_img_count = 1
key = 0

'''
def ImageDetection(location, index):
    new_index = index
    global total_img_count,key
    
    for i in file:
        #print(i)
        if(key==27):break
        if (i[-1:-4:-1]=='jpg'[::-1]):
            address = location+'/'+i
            try:           
                img = cv2.imread(address)        
                new_index = Detection(img, new_index)
                total_img_count+=1
            except :
                continue   
        else:
            continue
        
        #print(new_index)
        
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
            
    return new_index
'''
def Detection(ori_img, index):
    global total_img_count,key
    start=time.time()
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
    
    xList=[]
    yList=[]
    BoxS=[]
    BoxV=[]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
    #   
        
        rect = cv2.minAreaRect(c)

        rect = (rect[0], (w,h), 0)
        Box = np.int0(cv2.boxPoints(rect))
      
        
      
        #area1 = cv2.contourArea(Box)
        
        
        #print(Box)
        #print('\n')
        x=Box[1][0]
        y=Box[1][1]
        
        if (x<0 or x>(1280-w) or y<0 or y>(720-h)):continue
        
        flag=False
        for ix ,iy in zip(xList,yList):
            #計算重疊區域
            if ((((ix-x)**2+(iy-y)**2)**0.5)>150):continue #兩點>150時，判定為未重疊
            LeftTopx=max(x,ix)
            LeftTopy=max(y,iy) 
            if (LeftTopx==x):RightLowx=ix+w
            else:RightLowx=x+w
            if (LeftTopy==y):RightLowy=iy+h
            else:RightLowy=y+h
            Area=(RightLowx-LeftTopx)*(RightLowy-LeftTopy)
            if (Area/(w*h)>=0.4): #重疊區域>40%時，判定為重複框選
                print("Repeated area with %f"%(Area/(w*h)))
                flag=True
                break
        if flag :continue
        xList.append(x)
        yList.append(y)
        
        crop_img=ori_img[y:y+h,x:x+w]
        
        filename = "temp/temp.jpg"
        cv2.imwrite(filename,crop_img)#暫存圖片輸出
        #plt.imshow(crop_img)
        #print(filename)
                
        animal=predict.predict(filename)
        if (animal[0]=='星星'):
            BoxS.append(Box)
            #possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        elif (animal[0]=='武漢'):
            BoxV.append(Box)
            #possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 255, 0), 3)
        print(animal[0])
        #outputname='predict_'+str(total_img_count)
        index += 1
            
    end=time.time()
    print("Predict time:",end-start)
    #cv2.imwrite("testoutput/"+outputname+'.jpg',possible_img)
    #cv2.imshow(outputname, possible_img)
    #key=cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #print('(key:%d)'%key)
    

    return index,BoxS,BoxV


#ImageDetection(location, start_index)