import cv2
import os

# 選擇第攝影機
cap = cv2.VideoCapture(0)
# 設定影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 讀取資料夾資訊
file_path = 'E:\\training_data'
all_file = os.listdir(file_path)
pic_num = len(all_file)

while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('s'): # 若按下 s 鍵則拍照
        # 決定儲存圖片路徑與名稱
        pic_num = pic_num + 1
        output_path = file_path + '\\pic_' + str(pic_num) + '.jpg'
        # 儲存圖片
        cv2.imwrite(output_path, frame)
        # 拍照閃光效果，視覺輔助用
        #frame = cv2.convertScaleAbs(frame, alpha = 1, beta = 128)

    elif cv2.waitKey(1) & 0xFF == ord('q'): # 若按下 q 鍵則離開迴圈
        break
    
    # 顯示圖片
    cv2.imshow('frame', frame)

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()