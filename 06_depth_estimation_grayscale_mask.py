from ultralytics import YOLO
import cv2
import numpy as np

image = cv2.imread("images/image1.jpg")
model = YOLO("yolo26n-depth.pt")

results = model(image)
result = results[0]

# print(result)
# print(result.depth)
# print(result.depth.data.shape)

depth_map = result.depth.data

#so the actual depth data is stored in result.depth.data which is a pytorch tensor which we need to convert back into a numpy array
print("Before Conversion:", type(depth_map))
depth_map = depth_map.cpu().numpy()
print("After Conversion:", type(depth_map))

#lets visualize the datatype of the depth map, shape of the map, min value and max value
print("Depth map dtype",type(depth_map))
print("Depth map shape:", depth_map.shape)
print("Min value:", depth_map.min())
print("Max value:", depth_map.max())
# Output:
# Before Conversion: <class 'torch.Tensor'>
# After Conversion: <class 'numpy.ndarray'>
# Depth map dtype <class 'numpy.ndarray'>
# Depth map shape: (919, 736)
# Min value: 3.6291447
# Max value: 53.797325

# here our output has values like 3.6 and 53 all in meters. In order to plot these OpenCV expects an image representation
# so we have to normalize these number from 0 to 1 and then convert them to color values from 0 to 255

# cv2.normalize(src, dst, alpha, beta, norm_type)
# in our case we want to normalize the depth map from 0 to 255 so that we can visualize it as an image. We will use cv2.normalize function for this purpose.
# src : your depth_map
# dst : None
# alpha : 0
# beta : 255
# norm_type : cv2.NORM_MINMAX

depth_normalized = cv2.normalize (depth_map, None , 0 , 255, cv2.NORM_MINMAX).astype('uint8')
print("Normalized dtype:", depth_normalized.dtype)
print("Normalized shape:", depth_normalized.shape)
print("Normalized min:", depth_normalized.min())
print("Normalized max:", depth_normalized.max())

#lets make a coloured depth map without using open cvs colormap function and manually make the color mask
#we will use numpy vectorization to create a color map from the 0 to 255 normalized values in the depth map.


# first lets make an empty image as a blank canvas with same dimensions as the input image and 3 channels for RGB
colored_depth_map = np.zeros_like(image)

B = depth_normalized
G = depth_normalized
R = depth_normalized

colored_depth_map_list = [B, G, R]


print("Colored depth map list type:", type(colored_depth_map_list))
colored_depth_map = np.array(colored_depth_map_list)
print("Colored depth map type:", type(colored_depth_map))
print("Colored depth map shape:", colored_depth_map.shape)
print("input image shape", image.shape)
# output:
# Colored depth map shape: (3, 919, 736)
# input image shape (919, 736, 3)
# as we can see the dimensions are not in same order as the input image. The input image has shape (height, width, channels) but the colored depth map has shape (channels, height, width). So we need to transpose the colored depth map to match the input image shape.
# syntax of the transpose function is np.transpose(array, (new_order_of_dimensions)). In our case we want to change the order of dimensions from (channels, height, width) to (height, width, channels) so we will use np.transpose(colored_depth_map, (1, 2, 0)).
colored_depth_map = np.transpose(colored_depth_map, (1, 2, 0))

# syntax of addweighted function is cv2.addWeighted(src1, alpha, src2, beta, gamma). In our case we want the depth map on top of the image with more opacity for depth map 
output = cv2.addWeighted(image, 0.1, colored_depth_map , 0.9, 0)

cv2.imshow("Depth Map", output)
cv2.waitKey(0)
cv2.destroyAllWindows()