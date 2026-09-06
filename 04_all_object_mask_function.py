#nothing just made it into a custom function
from ultralytics import YOLO
import cv2
import numpy as np
import random

image = cv2.imread("images/image1.jpg")
model = YOLO("yolo11n-seg.pt") # So yolo is a class and i'm creating an object of this class called model

def Segmentation(image, model):
    
    # The inference will begin when i call this model object as a function
    results = model(image)  # i can enter multiple images by giving a python list of multiple images
    result = results[0] # result is a list with elements which can be accessed by indexing (similar to arrays in C++)

    #Result includes the objects it detected in the image, their masks, original image and many more data
    # since i have given one image only there is only one result which is stored in result[0] 
    # print(type(results[0]))

    # # visualizing the datatype of the mask created and the shape of the mask
    # print(type(result.masks))
    # print(type(result.masks.data))
    # print(result.masks.data.shape)

    # # This will print the mask with values between 1s and 0s 
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

        #lets get the class names from the model
        class_id = int( result.boxes.cls[i].item() ) #class id of each object detected
        print("Class ID: ",class_id)
        class_name = result.names[class_id]
        print("Class name: ",class_name) #name of the class 

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

        #now lets get the coordinates of the position of the mask
        coordinates = result.boxes.xyxy[i] #this gives x1,y1,x2,y2 of the bounding box
        x1 = coordinates[0].item() #We useitem() here because the putText function expects numeric values in the coordinates and result.boxes.xyxy[i] gives pytorch tensors
        y1 = coordinates[1].item()
        x2 = coordinates[2].item()
        y2 = coordinates[3].item()
        X = int((x1+x2)/2)
        Y = int((y1+y2)/2)
        position = [X,Y] # to display the text in the middle of the bounding box

        #Now lets put the text in the coordinate itself
        # cv2.putText(image, text, position, font, font_scale, color, thickness)
        cv2.putText(image, class_name, position ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2 )

        # image[mask_condition] = value
        colored_mask[Resized_mask > 0] = random_color() 

    # blend the mask with original image 
    # cv2.addWeighted(src1, alpha, src2, beta, gamma)
    # src1  : image
    # alpha : 0.7
    # src2  : colored_mask
    # beta  : 0.3
    # gamma : 0
    output = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    #code to make the output window size resizable
    if (image_height > 2160 or image_width > 3840):
        cv2.namedWindow("Segmentation Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Segmentation Output", 1200, 800)
    
    cv2.imshow("Segmentation Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Segmentation(image,model)
