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



# Global parameters
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP




def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "input_1", "*")))[:100]
    train_y = sorted(glob(os.path.join(path, "Categories_1", "*")))[:100]

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

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y
    
    image, mask = tf.numpy_function(f, [x, y], [np.float32, np.uint8])
    image.set_shape([IMG_H, IMG_W, 3])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])

    return image, mask

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":
    # Seeding the environment
    np.random.seed(41)
    tf.random.set_seed(41)

    #creating the dirctories
    create_dir("files")

    # Hyperparameters
    IMG_H = 640
    IMG_W = 832
    NUM_CLASSES = 18
    input_shape = (IMG_H, IMG_W, 3)
    batch_size = 8
    lr = 1e-4
    num_epoch = 30

    dataset_path = "data/"
    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "data.csv")

    # Loading the dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    # Colormap processing
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

    # Dataset pipeline
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # Model
    model = model.fcn8(input_shape=input_shape, num_classes=NUM_CLASSES)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )
    model.summary()

    # Training
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=num_epoch,
        callbacks=callbacks
    )
