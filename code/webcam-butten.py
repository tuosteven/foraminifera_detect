# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 00:41:50 2022

@author: acer
"""

import cv2
import os
import time
import predict
import numpy as np
import edge_detect_def as detect
# In[]
# 選擇第攝影機
cap = cv2.VideoCapture(1)
# 設定影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
new_index=0

boxflag=False
framecount=0
while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    if ((framecount%5)==0): # 每1幀做一次預測
        # 決定儲存圖片路徑與名稱
        output_path = 'temp/' + 'predictPic'+ '.jpg'
        # 儲存圖片
        cv2.imwrite(output_path, frame)
        
        img = cv2.imread(output_path)
        try:
            new_index,BoxS,BoxV = detect.Detection(img, new_index)
        except:print("nothing")
        if (boxflag==False):boxflag=True
        # 拍照閃光效果，視覺輔助用
        #frame = cv2.convertScaleAbs(frame, alpha = 1, beta = 128)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): # 若按下 q 鍵則離開迴圈
        break

    # 顯示圖片
    if boxflag:
        for Box in BoxS:
            frame = cv2.drawContours(frame, [Box], -1, (0, 0, 255), 3)
        for Box in BoxV:
            frame = cv2.drawContours(frame, [Box], -1, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    framecount+=1
    if(framecount==300000):framecount=0

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()