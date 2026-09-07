from ultralytics import YOLO
import cv2

image = cv2.imread("images/image1.jpg")
model = YOLO("yolo26n-depth.pt")

results = model(image)
result = results[0]

print(result)
print(result.depth)
print(result.depth.data.shape)