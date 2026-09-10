# You can skip all the nonsense i just wrote manually by using the .plot() function given by YOLO and make your life a bit less miserable

from ultralytics import YOLO
import cv2

image = cv2.imread("images/image1.jpg")

model = YOLO("yolo11n-seg.pt")

results = model (image)

annotated_image = results[0].plot()

cv2.imshow("YOLO", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()