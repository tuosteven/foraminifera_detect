# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 00:17:18 2021

@author: jacky
""" 

import cv2
#import ImageProcess
import numpy as np 

#截圖長寬 
w=150
h=150
# In[]:
ori_img = cv2.imread("image/pic_543.jpg")

cv2.imshow('img', ori_img)
img1=cv2.imshow('img', ori_img)


img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 10)
cv2.imshow('img1', img)

imgmax=img.max()

print(imgmax)
#XY = (1,1)
#Weight=(0.9,0.1,0)
#img_x = cv2.Sobel(img, cv2.CV_16S, XY[0], 0)
#img_y = cv2.Sobel(img, cv2.CV_16S, 0, XY[1])
#absX = cv2.convertScaleAbs(img_x)
#absY = cv2.convertScaleAbs(img_y)
#post_img=cv2.addWeighted(absX, Weight[0], absY, Weight[1], Weight[2])
#cv2.imshow('post_img111', post_img)




#ret, th1 = cv2.threshold(img3,0,255,cv2.THRESH_BINARY)

#post_img=ImageProcess.Edge_Detection(img,'Canny',th=(20,30))

#post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=7)
#ret, th1 = cv2.threshold(th1,1,255,cv2.THRESH_BINARY)
#cv2.imshow('post_img111', th1)
   


#cv2.imshow('post_img', post_img)
#cv2.imshow('th1_post_img', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('ouutput/lenna_final.jpg', th1)



# In[]:  
threshild_low=(imgmax*1.304)-145
print(threshild_low)


ret, th1 = cv2.threshold(img,threshild_low,255,cv2.THRESH_BINARY)
cv2.imshow('img', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[]:
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#執行影象形態學
closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
cv2.imshow('erode dilate', closed)
cv2.waitKey(0)
cv2.destroyAllWindows() 
# 腐蝕 4 次，去掉細節
closed = cv2.erode(closed, None, iterations=5)
cv2.imshow('erode dilate', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 膨脹 14 次，讓輪廓突出
closed = cv2.dilate(closed, None, iterations=2 )
cv2.imshow('erode dilate', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

(_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# In[]: determine by opencv
possible_img = ori_img.copy()
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect1 = cv2.minAreaRect(c)
rect = cv2.minAreaRect(c)
Box = np.int0(cv2.boxPoints(rect))
Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
area = cv2.contourArea(Box)
cv2.imshow('Final_img', Final_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()



# In[]: determine by opencv
possible_img = ori_img.copy()

for c in sorted(cnts, key=cv2.contourArea, reverse=True):
#  #print (c)
#  
    rect = cv2.minAreaRect(c)
#  #print(rect)
#  #print ('rectt', rect)
    rect = (rect[0], (w,h), 0)
    Box = np.int0(cv2.boxPoints(rect))
  
  
  #sumX=0
  #sumY=0
  #for i in range(4):
      #Box[i][0]
  #print ('Box', Box) 
  #Box=ImageProcess.order_points_new(Box) # return 左上/右上/右下/左下 (x,y)
  #print ('Box2',Box)
    area1 = cv2.contourArea(Box)
  #if   area1 <= area:
    possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    cv2.imshow('possible_img', possible_img)
#  
# #cv2.imshow('possible_img', possible_img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
# In[]:
cut_img = ori_img.copy()

d=0


for c in sorted(cnts, key=cv2.contourArea, reverse=True):
    rect = cv2.minAreaRect(c)
 #print(rect)
 #print ('rectt', rect)
    rect = (rect[0], (w,h), 0)
    Box = np.int0(cv2.boxPoints(rect))
    print(Box)
    print('\n')
    x=Box[1][0]
    y=Box[1][1]
    if (x<0 or x>(1280-w) or y<0 or y>(720-h)):continue

    crop_img=cut_img[y:y+h,x:x+w]
    
    filename = "output/%s_%d.jpg"%('first',d)
    
    cv2.imwrite(filename,crop_img)
    
    cv2.imshow("crop_img",crop_img)
    
    d=d+1
    #cv2.waitKey(0)
cv2.destroyAllWindows()