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
    def get_num(self):
        return self.input_img_path.split("/")[-1].split("_")[-1].split(".")[0]
            
    def get_prediction_vector(self):
        image_path = self.input_img_path
        mask_path = self.input_mask_path
        model = YOLO(self.model_path)
        result = model.predict(source= image_path, show =False)
        
        output_vector = {}
        for x in range(len(result[0].boxes.xyxy)):
            output_vector[int(result[0].boxes.cls[x])] =(result[0].boxes.xyxy[x])
        return output_vector
    
    def save_prediction_input(self, out_img_vector):
        map=["head","body"]
        image_num = self.get_num()
        map_path = ["data/input_head/img_", "data/input_body/img_"]
        # map_path = ["img_head", "img_body"]
        if len(out_img_vector) == 2:
            #we have two class prediction
            for i in range(2):
                vector = out_img_vector[i]
                orginal_image =cv2.imread(self.input_img_path)
                crop_file = orginal_image[int(vector[1]) :  int(vector[3]),  int(vector[0]) :int(vector[2] ) ]
                cv2.imwrite(f"{map_path[i]}{image_num}.jpg", crop_file)
                # cv2.imshow(f"{map[i]}.jpg", crop_file)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
        elif len(out_img_vector) ==1:
            #we have one class
            if list(out_img_vector.keys())[0] == 1:
                #vector at index 0 is body vector
                for i in range(1,2):
                    vector = out_img_vector[i]
                    orginal_image =cv2.imread(self.input_img_path)
                    crop_file = orginal_image[int(vector[1]) :  int(vector[3]),  int(vector[0]) :int(vector[2] ) ]
                    cv2.imwrite(f"data/input_body/img_{image_num}.jpg", crop_file)
                    # cv2.imwrite(f"img_{image_num}.jpg", crop_file)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                #wring a black head file
                cv2.imwrite(f"data/input_head/img_{image_num}.jpg", np.zeros((128, 128, 3), dtype=np.uint8))

            else:
                #vector at index 0 is head vector
                for i in range(0,1):
                    vector = out_img_vector[i]
                    orginal_image =cv2.imread(self.input_img_path)
                    crop_file = orginal_image[int(vector[1]) :  int(vector[3]),  int(vector[0]) :int(vector[2] ) ]
                    # cv2.imwrite(f"img_{image_num}.jpg", crop_file)
                    cv2.imwrite(f"data/input_head/img_{image_num}.jpg", crop_file)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                #writing a black body file
                cv2.imwrite(f"data/input_body/img_{image_num}.jpg", np.zeros((512, 512, 3), dtype=np.uint8))
        else:
            #none vector present
            #wring a black head and body file
            cv2.imwrite(f"data/input_head/img_{image_num}.jpg", np.zeros((128, 128, 3), dtype=np.uint8))
            cv2.imwrite(f"data/input_body/img_{image_num}.jpg", np.zeros((512, 512, 3), dtype=np.uint8))

    def save_prediction_input_mask(self, out_img_vector):
        map=["head","body"]
        image_num = self.get_num()
        map_path = ["data/output_head/img_", "data/output_body/img_"]
        # map_path = ["img_head", "img_body"]
        if len(out_img_vector) == 2:
            #we have two class prediction
            for i in range(2):
                vector = out_img_vector[i]
                orginal_image =cv2.imread(self.input_mask_path)
                crop_file = orginal_image[int(vector[1]) :  int(vector[3]),  int(vector[0]) :int(vector[2] ) ]
                cv2.imwrite(f"{map_path[i]}{image_num}.png", crop_file)
                # cv2.imshow(f"{map[i]}.jpg", crop_file)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
        elif len(out_img_vector) ==1:
            #we have one class
            if list(out_img_vector.keys())[0] == 1:
                #vector at index 0 is body vector
                for i in range(1,2):
                    vector = out_img_vector[i]
                    orginal_image =cv2.imread(self.input_mask_path)
                    crop_file = orginal_image[int(vector[1]) :  int(vector[3]),  int(vector[0]) :int(vector[2] ) ]
                    cv2.imwrite(f"data/output_body/img_{image_num}.png", crop_file)
                    # cv2.imwrite(f"img_{image_num}.jpg", crop_file)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                #wring a black head file
                cv2.imwrite(f"data/output_head/img_{image_num}.png", np.zeros((128, 128, 3), dtype=np.uint8))

            else:
                #vector at index 0 is head vector
                for i in range(0,1):
                    vector = out_img_vector[i]
                    orginal_image =cv2.imread(self.input_img_path)
                    crop_file = orginal_image[int(vector[1]) :  int(vector[3]),  int(vector[0]) :int(vector[2] ) ]
                    # cv2.imwrite(f"img_{image_num}.jpg", crop_file)
                    cv2.imwrite(f"data/output_head/img_{image_num}.png", crop_file)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                #writing a black body file
                cv2.imwrite(f"data/output_body/img_{image_num}.png", np.zeros((512, 512, 3), dtype=np.uint8))
        else:
            #none vector present
            #wring a black head and body file
            cv2.imwrite(f"data/output_head/img_{image_num}.png", np.zeros((128, 128, 3), dtype=np.uint8))
            cv2.imwrite(f"data/output_body/img_{image_num}.png", np.zeros((512, 512, 3), dtype=np.uint8))



        
    
    def get_vector_list(self):
        """
        returns the output list of predicted classes in sequence
        Args:
            -
        Return:
            di: Dict - with output classes corrdinates in xyxy form
        """
        model = YOLO(self.model_path)
        result = model.predict(source=self.input_img_path, show = False)

        output_file = {}
        for x in range(len(result[0].boxes.xyxy)):
            output_file[int(result[0].boxes.cls[x])] = result[0].boxes.xyxy[x]

        di = []
        if list(output_file.keys())[0] == 0:
            di = list(output_file.values())
        else:
            di = list(output_file.values())
            di = di[::-1]

        return di
    
    
        


    

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
