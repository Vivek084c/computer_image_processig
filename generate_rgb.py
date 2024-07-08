import cv2
import numpy as np

rgb_color_map=[[0,0,0], [0,0,128], [0,0,255], [0,85,0], [51,0,170], [0,85,255], [85,0,0], [221,119,0], [0,85,85], [85,85,0], [0,51,85], [128,86,52], [0,28,0], [255,0,0], [221,170,51], [225,225,0], [170,255,85], [85,255,170]]
classes=["Background",  "Hat",  "Hair",  "Sunglasses",  "Upper-clothes", "Skirt", "Pants",  "Dress", "Belt",  "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]


# def grayscale_to_rgb(mask_array, classes, rgb_color_map):
#     np.expand_dims(mask_array, axis=-1)
#     h, w= mask_array.shape
#     mask_array=mask_array.astype(np.int32)

#     output = []
#     for i, pixcel in enumerate(mask_array.flatten()):
#         output.append(rgb_color_map[pixcel])
#     output=np.reshape(output, (h,w,3))
#     return output

# for i in range(1000):
#     rgb= cv2.imread(f"data/Categories/img_{i}.png",cv2.IMREAD_ANYDEPTH)
#     rgb = grayscale_to_rgb(rgb, classes, rgb_color_map)
#     cv2.imwrite(f"data/Categories_1/img_{i}.png",rgb)



for k in range(5):
    x = cv2.imread(f"data/input/img_{k}.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("imga file", x)

    while True:
        key = cv2.waitKey(0)
        if key in [27, ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            break

print("success")

        
