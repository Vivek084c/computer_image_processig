from YOLO_Model import head_body_detection_model
import cv2


if __name__ == "__main__":
    for i in range(10000):

        if i%100 == 0 :
            print("-----------------",i,"--------------------------")
        input_image_path  =f"data/input/img_{i}.jpg"
        input_mask_path = f"data/Categories_1/img_{i}.png"
        model_path = "YOLO_Model/best.pt"

        model_1 = head_body_detection_model.HeadBodyClassifier(input_img_path=input_image_path, input_mask_path=input_mask_path, model_path = model_path)
        out_img_vector = model_1.get_prediction_vector()
        # model writen a null black image for input if not detected, and black for output mask
        
        model_1.save_prediction_input_mask(out_img_vector)
        
        



        
