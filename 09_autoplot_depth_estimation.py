# from ultralytics import YOLO
# import cv2

# image = cv2.imread("images/image1.jpg")
# model = YOLO("yolo26n-depth.pt")
# result = model(image)[0]

# result.show() # opens a window with the depth overlay
# # result.save("depth_overlay.png")  # or save it directly


# or you can do the entire thing using opencv for more control over the output:

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo26n-depth.pt")
image = cv2.imread("images/image1.jpg")

results = model(image)
result = results[0]

depth_map = result.depth.data.cpu().numpy()  # (H, W) float32, meters

# normalize to 0-255 uint8, same as before
depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# this single line replaces my entire manual blue_region/cyan_region/... block

colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
# colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
# colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO )
# colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
# colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)


# blend with original image, same as before
output = cv2.addWeighted(image, 0.1, colored_depth_map, 0.9, 0)

cv2.imshow("Depth Map", output)
cv2.waitKey(0)
cv2.destroyAllWindows()