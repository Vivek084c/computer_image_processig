import numpy as np
import tensorflow as tf
import os
import glob


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "input", "*")))[2500:3500]
    train_y = sorted(glob(os.path.join(path, "Categories_1", "*")))[2500:3500]

    # temp code start
    train_x,valid_x,test_x = train_x[:600], train_x[600:750], train_x[750:]
    train_y,valid_y,test_y = train_y[:600], train_y[600:750], train_y[750:]

    #temp code end

    # split_size = int(split * len(train_x))

    # train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    # train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    # train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    # train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

if __name__ == "__main__":
    """ seading the envriotnemnet"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """creating the directory"""
    create_dir("results")

    """directory to save result """
    ING_H = 512
    IMG_W = 512
    NUM_CLASSES = 18
    dataset_path = "data/"
    model_path  = os.path.join("files","model_1.keras")

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
    model =  tf.keras.model.load_model(model_path)

    """loading the dataset"""
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset()
    
    