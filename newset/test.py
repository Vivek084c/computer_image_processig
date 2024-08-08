import numpy as np
import tensorflow as tf
import os
import glob
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def grayscale_to_rgb(mask, classes, colormap):
    h, w, _ = mask.shape
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image, mask, pred, save_image_path):
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def load_dataset(path, split=0.2):
    train_x = sorted(glob.glob(os.path.join(path, "input",  "*")))[:5000]
    train_y = sorted(glob.glob(os.path.join(path, "Categories_1",  "*")))[:5000]

    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

# def load_dataset(path, split=0.2):
#     train_x = sorted(glob(os.path.join(path, "input", "*")))[:2500]
#     train_y = sorted(glob(os.path.join(path, "Categories_1", "*")))[:2500]

#     # temp code start
#     train_x,valid_x,test_x = train_x[0:1850], train_x[1850:2250], train_x[2250:]
#     train_y,valid_y,test_y = train_y[0:1850], train_y[1850:2250], train_y[2250:]

#     #temp code end

#     # split_size = int(split * len(train_x))

#     # train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
#     # train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

#     # train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
#     # train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

#     return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

if __name__ == "__main__":
    """ seading the envriotnemnet"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """creating the directory"""
    create_dir("results")

    """directory to save result """
    IMG_H = 512
    IMG_W = 512
    NUM_CLASSES = 18
    dataset_path = "data/"
    model_path  = os.path.join("newset","files","model_vivek.keras")

    """colormap"""
    COLORMAP = [
        [0, 0, 0], [0, 0, 128], [0, 0, 255], [0, 85, 0], [51, 0, 170],
        [0, 85, 255], [85, 0, 0], [221, 119, 0], [0, 85, 85], [85, 85, 0],
        [0, 51, 85], [128, 86, 52], [0, 28, 0], [255, 0, 0], [221, 170, 51],
        [225, 225, 0], [170, 255, 85], [85, 255, 170]
    ]
    CLASSES = [
        "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
        "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg",
        "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"
    ]

    """loading the model"""
    model = tf.keras.models.load_model(model_path)
    model.summary()

    """loading the dataset"""
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)



    """prediction and evaluation"""
    score = []
    c=0
    for x,y in tqdm(zip(test_x, test_y), total=len(train_x)):
        name  = x.split("/")[-1].split(".")[0].split("_")[-1]
        print(name)
        # data/input_head/img_0.jpg
        # data/output_head/img_0.png

        """reading the image file"""
        image = cv2.imread("data/input_head/img_0.jpg", cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMG_W, IMG_H))
        image_x = image
        image = image / 255.0
        image = np.expand_dims(image, axis=0)


        """reading the mask file"""
        mask = cv2.imread("data/output_head/img_0.png", cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (IMG_W, IMG_H))
        mask_x = mask
        onehot_mask = []
        for color in COLORMAP:
            cmap = np.all(np.equal(mask, color), axis=-1)
            onehot_mask.append(cmap)
        onehot_mask = np.stack(onehot_mask, axis=-1)
        onehot_mask = np.argmax(onehot_mask, axis=-1)
        onehot_mask = onehot_mask.astype(np.int32)

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred = np.argmax(pred, axis=-1)
        pred = pred.astype(np.float32)

        save_image_path = f"{name}.png"
        save_results(image_x, mask_x, pred, save_image_path)

        break

        """ Saving the prediction """

       

        
    
    
    