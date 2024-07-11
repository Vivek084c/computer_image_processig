from YOLO_Model import head_body_detection_model
import cv2


if __name__ == "__main__":
    for i in range(100):

        if i%100 == 0 :
            print("-----------------",i,"--------------------------")
        input_image_path  =f"data/input/img_{i}.jpg"
        input_mask_path = f"data/Categories_1/img_{i}.png"
        model_path = "YOLO_Model/best.pt"

        model_1 = head_body_detection_model.HeadBodyClassifier(input_img_path=input_image_path, input_mask_path=input_mask_path, model_path = model_path)
        out_img, out_mask = model_1.final_fin()
        # model writen a null black image for input if not detected, and black for output mask
        out_img=out_img["head"]
        out_mask=out_mask["head"]

        cv2.imwrite(f"data/input_head/img_{i}.jpg", out_img)
        cv2.imwrite(f"data/output_head/img_{i}.png", out_mask)



        

