# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:12:36 2022

@author: peter
"""

import cv2
cap = cv2.VideoCapture(1)
# 設定影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'): # 若按下 q 鍵則離開迴圈
        break
    cv2.imshow('frame', frame)
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()