from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt") # So yolo is a class and i'm creating an object of this class called model

# I guess the inference will begin when i call this model object as a function

results = model("images/image2.jpg")  # i can enter multiple images by giving a python list of multiple images

# result is a list with elements which can be accessed by indexing (similar to arrays in C++)
# since i have given one image only there is only one result which is stored in result[0] 

print(type(results[0]))
print(dir(results[0]))