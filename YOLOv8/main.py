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

def show_head_predicted_image(original_image_path, cordinates):
    original_image = cv2.imread(original_image_path)
    predicted_image = cv2.rectangle(original_image,  (int(cordinates[0]), int(cordinates[1])), (int(cordinates[2]), int(cordinates[3])), (0,255,0), 1)
    cv2.imshow("predicted image" , predicted_image)
    cv2.waitKey()

def display_head_body(original_image_path, result):
    # getting the head and body vector
    
    for i in result[0].names:
        original_image = cv2.imread(original_image_path)
        cv2.imshow("original_image_1", original_image)
        # print(i, result[0].names[i])
        print(result[0].boxes.xyxy[i], f"its the { result[0].names[i]} vector")
        cv2.rectangle(original_image,  (int(result[0].boxes.xyxy[i][0]), int(result[0].boxes.xyxy[i][1])), (int(result[0].boxes.xyxy[i][2]), int(result[0].boxes.xyxy[i][3])), (0,255,0), 1)
        cv2.imshow(f"predicted {result[0].names[i]} image", original_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        



if __name__ == "__main__":
    # setting up the model 
    print("start")
    model = YOLO("best.pt")

    # getting predition on specified image path
    image_path="dataset/img_1.jpg"
    result = model.predict(source=image_path, show=False)
    # using the xywh formate 
    head = result[0].boxes.xywh[0]
    body = result[0].boxes.xywh[1]
    original_img = cv2.imread(image_path)


    # cv2.rectangle(original_img , convert_xywh_to_xyxy(head), (0,255,0), 1)
    # cv2.imshow("head image" , original_img)
    # cv2.waitKey()

    # original_img = cv2.imread(image_path)
    # cv2.rectangle(original_img , convert_xywh_to_xyxy(body), (0,255,0), 1)
    # cv2.imshow("body image" , original_img)
    # cv2.waitKey()



    # result[0].save_txt("vivek.txt")
    # print(result[0].names)
    # print(result[0].boxes.xyxyn[0])
    # print(result[0].boxes.xyxyn[1])
    # head = result[0].boxes.xyxy[0]
    # body = result[0].boxes.xyxy[1]

    # cv2.rectangle(original_img,  (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), (0,255,0), 1)
    # cv2.imshow("head image" , original_img)
    # cv2.waitKey()

    # original_img = cv2.imread(image_path)
    # cv2.rectangle(original_img,  (int(body[0]), int(body[1])), (int(body[2]), int(body[3])), (0,255,0), 1)
    # cv2.imshow("body image", original_img)
    # cv2.waitKey()



