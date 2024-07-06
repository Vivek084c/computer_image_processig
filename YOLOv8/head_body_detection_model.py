#improting the libraries
from ultralytics import YOLO
import cv2
import numpy as np

def save_predition(original_image_path, result):
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


def show_img(path: str):
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


def show_img(path: np.ndarray):
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
 

def display_head_body_prediction(original_image_path, result):
    """
    dispaly the head and body predicted anotated images for each class
    Args: 
        original_image_path: path to the original input file
        result: prediction of the yolo model
    return:
        None
    """
    k=0
    for i in result[0].names:
        original_image = cv2.imread(original_image_path)
        cv2.rectangle(original_image,  (int(result[0].boxes.xyxy[i][0]), int(result[0].boxes.xyxy[i][1])), (int(result[0].boxes.xyxy[i][2]), int(result[0].boxes.xyxy[i][3])), (0,255,0), 1)
        cv2.imshow(f"predicted {result[0].names[int(result[0].boxes.cls[k])]} image", original_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        k+=1


if __name__ == "__main__":
    # setting up the model 
    model = YOLO("best.pt")
    
    #running prediction on the image
    image_path = "dataset/img_0.jpg"
    result = model.predict(source= image_path, show=False)

    #getting the head and body cropped image
    out_file = save_predition(image_path, result)
    head_prediction = out_file["head"]
    body_prediction = out_file["body"]

    print(head_prediction.shape)
    print(body_prediction.shape)

    #viewing the prediction
    show_img(head_prediction)
    show_img(body_prediction)

    




    # output_file = {}
    # for i in range(1):
    #     image_path=f"dataset/img_{i}.jpg"
    #     result = model.predict(source=image_path, show=False)
    #     display_head_body(image_path, result)    

    #     output_file =  save_predition(image_path, result)

    # print(output_file.keys())

    # head_img = output_file["head"]
    # body_img = output_file["body"]

    # print("----showing the head image------")
    # cv2.imshow("head", head_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # print("-----showing the body image ------")
    # cv2.imshow("body" , body_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # image_path=f"dataset/img_99.jpg"
    # print(int(image_path.split("_")[1].split(".")[0]))

        
   
   
   
    

# first elenemtn of head hight : 78
# first elenemtn of head width : 76
# first elenemtn of body hight : 454
# first elenemtn of body width : 154
# 83.5
# 78.4
# 460.7
# 175.7

