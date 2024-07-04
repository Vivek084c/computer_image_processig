#improting the libraries
from ultralytics import YOLO
import cv2

def make_prediction(imgae_path, model):
    return model.predict(imgae_path, show=False)

def get_head_body_cordinates(result):
    # {0: 'head', 1: 'body'}----> wrong (idk how) ----> follow :  {0: 'body', 1: 'head'}
    return result[0].boxes.xyxy[1],result[0].boxes.xyxy[0]

def show_img_from_path(path):
    cv2.imshow("original_image" , cv2.imread(path))
    cv2.waitKey()

def show_head_predicted_image(original_image_path, head_cordinates):
    original_image = cv2.imread(original_image_path)
    predicted_image = cv2.rectangle(original_image,  (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), (0,255,0), 1)
    cv2.imshow("predicted head image" , predicted_image)
    cv2.waitKey()


if __name__ == "__main__":
    # setting up the model 
    print("start")
    model = YOLO("best.pt")

    # getting predition on specified image path
    image_path="dataset/img_0.jpg"
    result = make_prediction(image_path, model)

    #extracting the head and body class cordinates
    head, body = get_head_body_cordinates(result)
   

    #displaying the orginal image ---> passing the original image
    show_img_from_path(image_path)

    # #displaying head prediction image
    # show_head_predicted_image(original_image_path= image_path, head_cordinates=head)
    
    
    #displaying body prediction image
    original_image =cv2.imread(image_path)
    predicted_image = cv2.rectangle(original_image,  (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), (0,255,0), 1)
    cv2.imshow("body prediction", predicted_image)
    cv2.waitKey()

