from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt") # So yolo is a class and i'm creating an object of this class called model

# The inference will begin when i call this model object as a function
results = model("images/image2.jpg")  # i can enter multiple images by giving a python list of multiple images
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

# Trying to convert the first mask into a boolean mask
mask = result.masks.data[0] #we take one 2d array representing one mask
Bool_mask = mask > 0.5 #any number greater than 0.5 = true, less than 0.5 = false so it created a new binary mask
print(Bool_mask)

#now lets visulaize one mask in the image itself
original_image = result.orig_img #original image data

# There is a problem now the shape of the arrays of the mask and original images are not the same:
print("Original image shape:", original_image.shape)
print("Mask shape:", mask.shape)
# Original image shape: (836, 1254, 3)
# Mask shape: torch.Size([448, 640])
# also original shape is a numpy array and mask is a pytorch tensor

# So we need to resize the mask for which we will use a function of open CV: cv2.resize(source, (width, height))
# NOTE!!!! you need to convert the pytorch tensor into a numpy array before passing into the resize function
# image.shape → height, width
# cv2.resize  → width, height
print(type(Bool_mask))
Bool_mask = Bool_mask.cpu().numpy();
print(type(Bool_mask))
# Resized_mask = cv2.resize(, (1254,836))