#improting the libraries
from ultralytics import YOLO
import cv2
import numpy as np
import os 

import head_body_detection_model


if __name__ == "__main__":
    image_path  = "data/input/img_17.jpg"
    mask_path = "data/Categories_1/img_17.png"
    model  = YOLO("YOLO_Model/best.pt")
    result = model.predict(source= image_path, show = False)


    # print(f"path: {image_path}",  result[0].boxes.cls)
    # i=0
    # output_img = {}
    # output_mak = {}
    # for k in result[0].boxes.cls:
    #     corrosponding_vector = result[0].boxes.xyxy[int(k)]
    #     # print(int(k), corrosponding_vector)
    #     orignal_iamge = cv2.imread(image_path)
    #     mask_image = cv2.imread(mask_path)
    #     file = orignal_iamge[int(result[0].boxes.xyxy[i][1]) :  int(result[0].boxes.xyxy[i][3]),  int(result[0].boxes.xyxy[i][0]) :int(result[0].boxes.xyxy[i][2] ) ]
    #     mask = mask_image[int(result[0].boxes.xyxy[i][1]) :  int(result[0].boxes.xyxy[i][3]),  int(result[0].boxes.xyxy[i][0]) :int(result[0].boxes.xyxy[i][2] ) ]
    #     name = f"{result[0].names[int(k)]}"
    #     output_img[name] = file
    #     output_mak[name] = mask
    #     # cv2.imshow(name, mask)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #     i+=1
    map = ["head","body"]
    for k in range(0,50):
        image_path  = f"data/input/img_{k}.jpg"
        mask_path = f"data/Categories_1/img_{k}.png"
        model  = YOLO("YOLO_Model/best.pt")
        result = model.predict(source= image_path, show = False)
        # print(f"result vectors: {result[0].boxes.xyxy}")
        # print(f"classes available: {result[0].boxes.cls}")

        outptu_file ={}
        for x in range(len(result[0].boxes.xyxy)):
            outptu_file[int(result[0].boxes.cls[x])] =(result[0].boxes.xyxy[x])
    di = []

    if list(outptu_file.keys())[0] == 0:
        di = list(outptu_file.values())
    else:
        di = list(outptu_file.values())
        di = di[::-1]
    print(len(di))
        
    



        #displaying the files
        
            

    # print(f"lenghth before {len(output_img)} , {len(output_mak)}")

    # if len(output_img.keys()) !=2:
    #         if not "head" in output_img.keys():
    #             #head is not detected, we append a dummy head vector
    #             output_img["head"] = np.zeros((128, 128, 3), dtype=np.uint8)
    #         else:
    #             pass

    #         if not "body" in output_img.keys():
    #             #body is not detected, we append a dummy body vector
    #             output_img["body"]  =  np.zeros((512, 512, 3), dtype=np.uint8)
    #         else:
    #             pass
    # else:
    #         pass

    # if len(output_mak.keys()) !=2:
    #         if not "head" in output_mak.keys():
    #             #head is not detected, we append a dummy head vector
    #             output_mak["head"] =  np.zeros((128, 128, 3), dtype=np.uint8)
    #         else:
    #             pass

    #         if not "body" in output_mak.keys():
    #             #body is not detected, we append a dummy body vector
    #             output_mak["body"]  = np.zeros((512, 512, 3), dtype=np.uint8)
    #         else:
    #             pass
    # else:
    #         pass

    # print(f"lenghth after {len(output_img)} , {len(output_mak)}")


    # cv2.imshow("head" , output_img["head"])
    # cv2.imshow("body",output_img["body"])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow("head" , output_mak["head"])
    # cv2.imshow("body",output_mak["body"])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    

        