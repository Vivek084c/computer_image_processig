from YOLO_Model import head_body_detection_model
from ultralytics import YOLO
import cv2
import numpy as np



if __name__ == "__main__":
    #input file path
    input_image_path = "data/input_1/img_0.jpg"
    input_mask_path = "data/Categories_1/img_0.png"
    model_path = "YOLO_Model/best.pt"

    #defining a model class for head and body detection
    model_1 = head_body_detection_model.HeadBodyClassifier(input_img_path=input_image_path, input_mask_path=input_mask_path, model_path = model_path)

    #getting the head and body vector
    out_image_file, out_mask_fiile= model_1.get_head_body_vector()

    #extracting head and body vector
    head_img = out_image_file["head"]
    body_img = out_image_file["body"]

    # displaying the head and body vector
    model_1.show_img(head_img)
    model_1.show_img(body_img)

    #iextracting head and body mask vector
    head_img_mask = out_mask_fiile["head"]
    body_img_mask = out_mask_fiile["body"]

    #displaying the mask
    model_1.show_img(head_img_mask)
    model_1.show_img(body_img_mask)


  

    



