#improting the libraries
from ultralytics import YOLO
import cv2

def make_prediction(imgae_path, model):
    return model.predict(imgae_path, show=False)

def get_head_body_cordinates(result):
    # {0: 'head', 1: 'body'}
    head_cordinates= result[0].boxes.xyxy[0]
    body_corrdinates=result[0].boxes.xyxy[1]
    return head_cordinates, body_corrdinates

if __name__ == "__main__":
    # setting up the model 
    print("start")
    model = YOLO("dataset/best.pt")

    # getting predition on specified image path
    image_path="dataset/img_0.jpg"
    result = make_prediction(image_path, model)

    #extracting the head and body class cordinates
    head, body = get_head_body_cordinates(result)
    print(f"the head cordinates are: {head}")
    print(f"the body cordinates are : {body}")

    #displaying the orginal image
    

