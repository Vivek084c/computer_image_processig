import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split

class Utils:
    def __init__(self, IMG_H, IMG_W, COLORMAP):
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.COLORMAP = COLORMAP
        

def create_dir(path):
    """
    create necessary directory 
    Args:
        path:str -> path to create directory
    return:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "input_1", "*")))[:300]
    train_y = sorted(glob(os.path.join(path, "Categories_1", "*")))[:300]

    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))  # Resize to (IMG_W, IMG_H)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))  # Resize to (IMG_W, IMG_H)

    # Processing the mask
    output = []
    for color in COLORMAP:
        cmap = np.all(np.equal(x, color), axis=-1)
        output.append(cmap)
    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)
    return output
