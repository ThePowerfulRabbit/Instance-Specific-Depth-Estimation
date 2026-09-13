#nothing just made it into a custom function
from ultralytics import YOLO
import cv2
import numpy as np
import random

image = cv2.imread("images/image1.jpg")
image_for_display = image.copy() #making a copy of the original image for segmentation
segmentation_model = YOLO("yolo11n-seg.pt")
depth_model = YOLO("yolo26n-depth.pt")

# making the depth map in meters
depth_map_result = depth_model(image)
depth_map = depth_map_result[0].depth.data
depth_map = depth_map.cpu().numpy()

def random_color(): #color generator for different masks
        B = random.randint(0,255)
        G = random.randint(0,255)
        R = random.randint(0,255)
        color =[B,G,R]
        return color

def Instance_depth_estimation(image, model, depth_map):
    results = model(image, retina_masks = True)  # i can enter multiple images by giving a python list of multiple images
    result = results[0] # result is a list with elements which can be accessed by indexing (similar to arrays in C++)

    image_height, image_width, channels = image.shape #original image dimensions

    #getting the total number of masks/objects the model detected
    no_objects = result.masks.data.shape[0]

    #creating empty image with same dimension as original image which will be the canvas for masks
    segmentation_mask = np.zeros_like(image)

    for i in range (0, no_objects):
        #lets get the class names from the model
        class_id = int( result.boxes.cls[i].item() ) #class id of each object detected
        # print("Class ID: ",class_id)
        class_name = result.names[class_id]
        # print("Class name: ",class_name) #name of the class 
        
        # Trying to convert the first mask into a boolean mask
        num_mask = result.masks.data[i] #we take one 2d array representing one mask
        
        BW_mask = (num_mask > 0.5) * 255 #any number greater than 0.5 = true * 255 = 255, less than 0.5 = false = 0 *255 = 0 so it created a new mask
        BW_mask = BW_mask.cpu().numpy().astype('uint8'); #pytorch to numpy , also typecasting int16 to uint8 because the resize function does not accept int16
        # print(type(BW_mask))
        
        depth_map_boolean = depth_map[BW_mask == 255] #using vectorization for selecting only the pixels with the masks for getting a 1D array of the respective pixel values
        object_depth_avg = np.mean(depth_map_boolean) # calculating average value of all numbers
        depth_info = f"{object_depth_avg:.2f} m"  #converting the number to string and rounding upto 2 decimal places

        Resized_mask = cv2.resize(BW_mask, (image_width, image_height))

        coordinates = result.boxes.xyxy[i] #this gives x1,y1,x2,y2 of the bounding box
        x1 = coordinates[0].item() #We useitem() here because the putText function expects numeric values in the coordinates and result.boxes.xyxy[i] gives pytorch tensors
        y1 = coordinates[1].item()
        x2 = coordinates[2].item()
        y2 = coordinates[3].item()
        X = int((x1+x2)/2)
        Y = int((y1+y2)/2)
        class_name_position = [X,Y] # to display the text in the middle of the bounding box
        depth_position = [X, Y+20]

        cv2.putText(image, class_name, class_name_position ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2 )
        cv2.putText(image, depth_info , depth_position ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2 )

        segmentation_mask[Resized_mask > 0] = random_color() #apply a random color to pixels in segmentation_mask with corresponding pixels in Resized_mask greater than 0
        
    return segmentation_mask, image_width, image_height

# making segmentation mask and bending the colored msk with the input image
segmentation_mask, image_width, image_height  = Instance_depth_estimation(image_for_display, segmentation_model, depth_map)
segmentation_output = cv2.addWeighted(image_for_display, 0.7, segmentation_mask, 0.3, 0)

#code to make the output window size resizable
if (image_height > 2160 or image_width > 3840):
    cv2.namedWindow("Instance specific depth estimation Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Instance specific depth estimation Output", 1200, 800)

cv2.imshow("Instance specific depth estimation Output", segmentation_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
