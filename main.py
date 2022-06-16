# encoding=utf-8
# Programmer: 410929585， 魏正權
# 期末報告
# 人臉圖片檢測

# 載入 dlib 套件
# 載入 cv2 套件
# 載入 imutils 套件
import dlib
import cv2
import imutils

# 匯入圖片
# 給予兩個不同的名稱，因為要展示 before 和 after 的差別
selected_img = 'three_of_us.jpg'

# 原本和修改尺寸的圖片初始化
imageOriginal = cv2.imread(selected_img)
imageToRestructure = cv2.imread(selected_img)

# 利用 imutils 把原本的圖片放大
imageToRestructure = imutils.resize(imageToRestructure, width=1080, height=800)

# 利用 dlib 內建的人臉識別 function
detector = dlib.get_frontal_face_detector()

# 列印圖片的資料型態和影像的維度
# 會列印 imageOriginal 和 imageToRestructure
print("imageOriginal 影像的資料型態: ", type(imageOriginal))
print("imageOriginal 影像的維度: ", imageOriginal.shape)
print()
print("imageToRestructure 影像的資料型態: ", type(imageToRestructure))
print("imageToRestructure 影像的維度: ", imageToRestructure.shape)

# 偵測人臉，輸出分數
face_rects, scores, idx = detector.run(imageToRestructure, 0, 0)

# 利用 for 迴圈 在圖片抓取像是人臉的物件
# 然後將其物件利用綠色框，框起來
# 將其物件的輸出人臉的分數在綠色框的上方
image_detected = 0
for i, d in enumerate(face_rects):
    rectLeft = d.left()
    rectTop = d.top()
    rectRight = d.right()
    rectBottom = d.bottom()
    show_text = "%5f(%d)" % (scores[i], idx[i])

    cv2.rectangle(imageToRestructure, (rectLeft, rectTop), (rectRight, rectBottom)
                  , (0, 300, 0), 2, cv2.LINE_AA)  # 設計方塊大小和顔色

    cv2.putText(imageToRestructure, show_text, (rectLeft, rectTop),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (300, 300, 300),
                1, cv2.LINE_AA)
    image_detected += 1
print(f"目前在當前照片檢測到的人臉一共有 {image_detected} 位")

# 建立兩個不同的視窗
# 分別為 imageOriginal 和 imageToRestructure
# 方便做對比
cv2.imshow("Original", imageOriginal)
cv2.imshow("Face Detection", imageToRestructure)

# 等待使用者在鍵盤點擊任何鍵盤鍵 輸出
cv2.waitKey(0)

# 當收到鍵盤輸入即關閉所有視窗
cv2.destroyAllWindows()
