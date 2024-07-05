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

def map(result):
    """
    it will take result vectore and return head and body vector
    """
    map_data = {}
    for i in range(2):
        temp = []
        class_name = result[0].name(int(result[0].boxes.cls[i]))
        temp.append(class_name)
        class_vector = result[0].boxes.xyyx[i]
        temp.append(class_vector)
        map_data[i] = temp
    return map_data
        

def display_head_body(original_image_path, result):
    # # getting the head and body vector
    # a, b=int(result[0].boxes.cls[0]), int(result[0].boxes.cls[1])
    # print(f"the valeu of a : {a} and the value of b : {b}")
    
    #setting up rectangle box 
    k=0
    for i in result[0].names:
        original_image = cv2.imread(original_image_path)
        cv2.imshow("original_image_1", original_image)
        # print(i, result[0].names[i])
        print(result[0].boxes.xyxy[i], f"its the { result[0].names[i]} vector and the value of i is :{i}")
        cv2.rectangle(original_image,  (int(result[0].boxes.xyxy[i][0]), int(result[0].boxes.xyxy[i][1])), (int(result[0].boxes.xyxy[i][2]), int(result[0].boxes.xyxy[i][3])), (0,255,0), 1)
        cv2.imshow(f"predicted {result[0].names[int(result[0].boxes.cls[k])]} image", original_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        k+=1


    # for i, _ in enumerate(map(result)):
    #     original_image = cv2.imread(original_image_path)
    #     cv2.rectangle(original_image, ( ((_[1][0]), (_[1][1])), ((_[1][2]), (_[1][3])) ) ,  (0,255,0), 1)
    #     cv2.imshow(f"predicted {_[0]} image", original_image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
        


        



if __name__ == "__main__":
    # setting up the model 
    print("start")
    model = YOLO("best.pt")
    for k in range(5):
        # getting predition on specified image path
        image_path=f"dataset/img_{k}.jpg"
        result = model.predict(source=image_path, show=False)
        
        display_head_body(image_path, result)
   
   
   
    



