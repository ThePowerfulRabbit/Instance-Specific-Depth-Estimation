from ultralytics import YOLO
import cv2
import numpy as np
import random

model = YOLO("yolo11n-seg.pt") # So yolo is a class and i'm creating an object of this class called model
image = cv2.imread("images/image1.jpg")
# The inference will begin when i call this model object as a function
results = model(image)  # i can enter multiple images by giving a python list of multiple images
result = results[0] # result is a list with elements which can be accessed by indexing (similar to arrays in C++)

#Result includes the objects it detected in the image, their masks, original image and many more data
# # since i have given one image only there is only one result which is stored in result[0] 
# print(type(results[0]))

# # visualizing the datatype of the mask created and the shape of the mask
# print(type(result.masks))
# print(type(result.masks.data))
# print(result.masks.data.shape)

# # This will print the mask with 1s and 0s and some floating point values in between 1 means the part of the image which has the particular object
# print(result.masks.data[0])
# print(result.masks.data[0].shape)

original_image = result.orig_img #original image data
image_height, image_width, channels = image.shape #original image dimensions

#getting the total number of masks/objects the model detected
no_objects = result.masks.data.shape[0]

#creating empty image with same dimension as original image which will b e the canvas for masks
colored_mask = np.zeros_like(image)

def random_color(): #color generator for different masks
    B = random.randint(0,255)
    G = random.randint(0,255)
    R = random.randint(0,255)
    color =[B,G,R]
    return color

for i in range (0, no_objects):
    # Trying to convert the first mask into a boolean mask
    num_mask = result.masks.data[i] #we take one 2d array representing one mask
    BW_mask = (num_mask > 0.5) * 255 #any number greater than 0.5 = true * 255 = 255, less than 0.5 = false = 0 *255 = 0 so it created a new mask
    # print(BW_mask)

    # There is a problem now the shape of the arrays of the mask and original images are not the same:
    print("Original image shape:", original_image.shape)
    print("Mask shape:", num_mask.shape)
    # Original image shape: (836, 1254, 3)
    # Mask shape: torch.Size([448, 640])
    # also original shape is a numpy array and mask is a pytorch tensor

    # So we need to resize the mask for which we will use a function of open CV: cv2.resize(source, (width, height))
    # NOTE!!!! you need to convert the pytorch tensor into a numpy array before passing into the resize function
    # image.shape → height, width
    # cv2.resize  → width, height
    print(type(BW_mask))
    BW_mask = BW_mask.cpu().numpy().astype('uint8'); #pytorch to numpy , also typecasting int16 to uint8 because the resize function does not accept int16
    print(type(BW_mask))

    # Resized_mask = cv2.resize(BW_mask, (1254,836)) : not recommended to hardcode the width and height
    Resized_mask = cv2.resize(BW_mask, (image_width, image_height))
    print(Resized_mask.shape)

    # image[mask_condition] = value
    colored_mask[Resized_mask > 0] = random_color() #Wherever Resized_mask > 0 is true, set the corresponding pixel in colored_mask to green.open cv uses [B,G,R]

# blend the mask with original image 
# cv2.addWeighted(src1, alpha, src2, beta, gamma)
# src1  : image
# alpha : 0.7
# src2  : colored_mask
# beta  : 0.3
# gamma : 0
output = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

cv2.imshow("Segmentation Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()