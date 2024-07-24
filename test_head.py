import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import model
from train import load_dataset
from tqdm import tqdm

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_results(image, mask, pred, save_image_path):
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


# Global parameters
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP


if __name__ == "__main__":
    #seading the environment
    np.random.seed(42)
    tf.random.set_seed(42)

    #creating directory
    create_dir("results")

    """hyper parameters"""
    IMG_H = 256
    IMG_W = 256
    NUM_CLASSES = 5
    dataset_path =  "data/"
    model_path = os.path.join("files", "model_head.keras")

    COLORMAP = [
        [0, 0, 0], [0, 0, 128], [0, 0, 255], [0, 85, 0], [128, 86, 52]
    ]
    CLASSES = [
        "Background", "Hat", "Hair", "Sunglasses", "Face"
    ]

    #loading the model
    model = tf.keras.models.load_model(model_path)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"train: {len(train_x)}/{len(train_y)} test: {len(test_x)}/{len(test_y)} valid: {len(valid_x)}/{len(valid_y)}")
    print("")

    # model.summary()


    """prediction and evaluation"""
    score = []
    c=0
    for x,y in tqdm(zip(test_x, test_y), total=len(train_x)):
        name  = x.split("/")[-1].split(".")[0].split("_")[-1]
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


        """ Saving the prediction """
        save_image_path = f"results/{name}.png"
        save_results(image_x, mask_x, pred, save_image_path)

        

        break
        # c+=1;
        # if c==7:
        #     break
