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
        

    

    #function to handle the above vector generation when head or body not detected
    def handle_not_detection(self, output_image: dict, output_mask):
        
        #handling the not output_image detection
        if len(output_image.keys()) !=2:
            if not "head" in output_image.keys():
                #head is not detected, we append a dummy head vector
                output_image["head"] = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                pass

            if not "body" in output_image.keys():
                #body is not detected, we append a dummy body vector
                output_image["body"]  =  np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                pass
        else:
            pass

        if len(output_mask.keys()) !=2:
            if not "head" in output_mask.keys():
                #head is not detected, we append a dummy head vector
                output_mask["head"] =  np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                pass

            if not "body" in output_mask.keys():
                #body is not detected, we append a dummy body vector
                output_mask["body"]  = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                pass
        else:
            pass

        return output_image, output_mask
            
    def final_fin(self):
        image_path = self.input_img_path
        mask_path = self.input_mask_path
        model = YOLO(self.model_path)
        result = model.predict(source= image_path, show =False)
        
        i=0
        output_img = {}
        output_mak = {}
        for k in result[0].boxes.cls:
            corrosponding_vector = result[0].boxes.xyxy[int(k)]
            # print(int(k), corrosponding_vector)
            orignal_iamge = cv2.imread(image_path)
            mask_image = cv2.imread(mask_path)
            file = orignal_iamge[int(result[0].boxes.xyxy[i][1]) :  int(result[0].boxes.xyxy[i][3]),  int(result[0].boxes.xyxy[i][0]) :int(result[0].boxes.xyxy[i][2] ) ]
            mask = mask_image[int(result[0].boxes.xyxy[i][1]) :  int(result[0].boxes.xyxy[i][3]),  int(result[0].boxes.xyxy[i][0]) :int(result[0].boxes.xyxy[i][2] ) ]
            name = f"{result[0].names[int(k)]}"
            output_img[name] = file
            output_mak[name] = mask
            # cv2.imshow(name, mask)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            i+=1
        output_img, output_mak = self.handle_not_detection(output_img, output_mak)
        return output_img, output_mak


    def get_head_body_vector(self):
        #loading the model
        model = YOLO(self.model_path)
        self.result = model.predict(source=self.input_img_path, show=False, conf=0.15)
        

        output_image ={}
        output_mask={}
        k = 0
        for i in self.result[0].names:
            temp = []
            original_image = cv2.imread(self.input_img_path)
            mask_image = cv2.imread(self.input_mask_path)
            _ = original_image[int(self.result[0].boxes.xyxy[i][1]) :  int(self.result[0].boxes.xyxy[i][3]),  int(self.result[0].boxes.xyxy[i][0]) :int(self.result[0].boxes.xyxy[i][2] ) ]
            __ = mask_image[int(self.result[0].boxes.xyxy[i][1]) :  int(self.result[0].boxes.xyxy[i][3]),  int(self.result[0].boxes.xyxy[i][0]) :int(self.result[0].boxes.xyxy[i][2] ) ]
            output_image[self.result[0].names[int(self.result[0].boxes.cls[k])]] = _
            output_mask[self.result[0].names[int(self.result[0].boxes.cls[k])]] = __
            k+=1
        
        return self.handle_not_detection( output_image, output_mask)
    

    def get_head_body_image_mask(self):
        #defining the model
        model = YOLO(self.model_path)
        self.result = model.predict(source=self.input_img_path, show=False, conf=0.15)

        output_image ={}
        output_mask={}

        result =self.result[0]
        i=0
        for k in  result:
            if i<len(result.boxes.xyxy):
                #getting the original images
                original_image = cv2.imread(self.input_img_path)
                mask_image = cv2.imread(self.input_mask_path)

                #getting the cropped version
                img = original_image[int(self.result[0].boxes.xyxy[i][1]) :  int(self.result[0].boxes.xyxy[i][3]),  int(self.result[0].boxes.xyxy[i][0]) :int(self.result[0].boxes.xyxy[i][2] ) ]
                mask = mask_image[int(self.result[0].boxes.xyxy[i][1]) :  int(self.result[0].boxes.xyxy[i][3]),  int(self.result[0].boxes.xyxy[i][0]) :int(self.result[0].boxes.xyxy[i][2] ) ]
                
                output_image[result.names[k]] = img
                output_mask[result.names[k]] = mask
                i+=1
            else:
                break
        return output_image, output_mask

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
