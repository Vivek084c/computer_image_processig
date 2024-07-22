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

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



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

    model.summary()


    """prediction and evaluation"""
    score = []
    for x,y in tqdm
