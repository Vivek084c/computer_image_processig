#improting the libraries
from ultralytics import YOLO
import cv2
import numpy as np

class HeadBodyClassifier:
    def __init__(self, input_img_path, input_mask_path, model_path):
        self.input_img_path = input_img_path
        self.input_mask_path = input_mask_path
        self.model_path = model_path
        self.result = None 
        

    def get_head_body_vector(self):
        #loading the model
        model = YOLO(self.model_path)
        self.result = model.predict(source=self.input_img_path, show=False)
        

        output_iamge ={}
        output_mask={}
        k = 0
        for i in self.result[0].names:
            temp = []
            original_image = cv2.imread(self.input_img_path)
            mask_image = cv2.imread(self.input_mask_path)
            PermissionError(mask_image.shape)
            _ = original_image[int(self.result[0].boxes.xyxy[i][1]) :  int(self.result[0].boxes.xyxy[i][3]),  int(self.result[0].boxes.xyxy[i][0]) :int(self.result[0].boxes.xyxy[i][2] ) ]
            __ = mask_image[int(self.result[0].boxes.xyxy[i][1]) :  int(self.result[0].boxes.xyxy[i][3]),  int(self.result[0].boxes.xyxy[i][0]) :int(self.result[0].boxes.xyxy[i][2] ) ]
            output_iamge[self.result[0].names[int(self.result[0].boxes.cls[k])]] = _
            output_mask[self.result[0].names[int(self.result[0].boxes.cls[k])]] = __
            k+=1
        return output_iamge, output_mask
    


    def save_predition(self):
        """
        return head and body prediction for an image
        Args: 
            original_image_path- original path of input image
            result- prediction of yolo model
        return: 
            output- dictionary with numpy array of the head and body cropped images
        """

        output ={}
        k = 0
        for i in result[0].names:
            temp = []
            original_image = cv2.imread(original_image_path)
            _ = original_image[int(result[0].boxes.xyxy[i][1]) :  int(result[0].boxes.xyxy[i][3]),  int(result[0].boxes.xyxy[i][0]) :int(result[0].boxes.xyxy[i][2] ) ]
            output[result[0].names[int(result[0].boxes.cls[k])]] = _
            k+=1
        return output


    def show_img(self, path: str):
        """
        show the image file from the given file path
        Args: 
            path: path to image file
        return: 
            None
        """
        cv2.imshow("original_image" , cv2.imread(path))
        cv2.waitKey()
        cv2.destroyAllWindows()


    def show_img(self, path: np.ndarray):
        """
        show the image file from the given numpy arrat
        Args: 
            path: numpy array to process the image
        return:
            None
        """
        cv2.imshow("prediction_componenet" , path)
        cv2.waitKey()
        cv2.destroyAllWindows()
 

    def display_head_body_prediction(self):
        """
        dispaly the head and body predicted anotated images for each class
        Args: 
            original_image_path: path to the original input file
            result: prediction of the yolo model
        return:
            None
        """
        k=0
        for i in self.result[0].names:
            original_image = cv2.imread(self.original_image_path)
            cv2.rectangle(original_image,  (int(self.result[0].boxes.xyxy[i][0]), int(self.result[0].boxes.xyxy[i][1])), (int(self.result[0].boxes.xyxy[i][2]), int(self.result[0].boxes.xyxy[i][3])), (0,255,0), 1)
            cv2.imshow(f"predicted {self.result[0].names[int(self.result[0].boxes.cls[k])]} image", original_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            k+=1
# first elenemtn of head hight : 78
# first elenemtn of head width : 76
# first elenemtn of body hight : 454
# first elenemtn of body width : 154
# 83.5
# 78.4
# 460.7
# 175.7


# --------- how to use yolo module ----------
# input_data_path = "data/input_1/img_991.jpg"
# model_path = "YOLO_Model/best.pt"
# model_1 = head_body_detection_model.HeadBodyClassifier(input_img_path=input_data_path, model_path = model_path)
# out_file = model_1.get_head_body_vector()
# head_img = out_file["head"]
# body_img = out_file["body"]

# head_body_detection_model.show_img(head_img)
# head_body_detection_model.show_img(body_img)
