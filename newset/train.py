# important modules
import os
import glob as glb
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


#model import 
import tensorflow as tf
from tensorflow.keras import layers, models




# Global parameters
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP


# important functions
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    train_x = sorted(glb(os.path.join(path, "input",  "*")))[:5000]
    train_y = sorted(glb(os.path.join(path, "Categories_1",  "*")))[:5000]

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


def build_fcn16(input_shape=(512, 512, 3), num_classes=18):
    inputs = layers.Input(shape=input_shape)
    
    # VGG16 Encoder
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg_layers = dict([(layer.name, layer.output) for layer in vgg.layers])

    # Use specific layers from VGG16 for skip connections
    pool3 = vgg_layers['block3_pool']
    pool4 = vgg_layers['block4_pool']
    pool5 = vgg_layers['block5_pool']

    # Fully convolutional layers
    x = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(pool5)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x)
    
    # Deconvolution layers
    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1), activation=None, padding='same')(pool4)])
    
    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1), activation=None, padding='same')(pool3)])
    
    x = layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    
    # Output layer
    outputs = layers.Activation('softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# # Create the model
# model = fcn16(input_shape=(512, 512, 3), num_classes=18)
# model.summary()



if __name__ == "__main__":
    #seading the environment
    # np.random.seed(41)
    # tf.random.set_seed(42)

    # #creating the directory
    # create_dir(os.path.join("newset", "files"))

    # #hyperparameter
    # IMG_H = 512
    # IMG_W = 512
    # NUM_CLASSES = 18
    # input_shape = (IMG_H, IMG_W, 3)
    # batch_size = 10
    # lr = 1e-4
    # num_epoch = 30

    #important paths
    dataset_path = "data/"
    model_path = os.path.join("newset","files", "model_head.keras")
    csv_path = os.path.join("newset","files", "data_head.csv")

    print(os.getcwd())
    l = os.path.join(dataset_path, "input","*")
    for k in l:
        print(k)
         

    # # Loading the dataset
    # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)


    #  # Colormap processing 
    # COLORMAP = [
    #     [0, 0, 0], [0, 0, 128], [0, 0, 255], [0, 85, 0], [51, 0, 170],
    #     [0, 85, 255], [85, 0, 0], [221, 119, 0], [0, 85, 85], [85, 85, 0],
    #     [0, 51, 85], [128, 86, 52], [0, 28, 0], [255, 0, 0], [221, 170, 51],
    #     [225, 225, 0], [170, 255, 85], [85, 255, 170]
    # ]
    # CLASSES = [
    #     "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    #     "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg",
    #     "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"
    # ]

    # # Dataset pipeline
    # train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    # valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)



    # # Model
    # model = fcn16(input_shape=input_shape, num_classes=NUM_CLASSES)
    # # sdg optimizer
    # sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, decay=0.0005)


    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=sgd
    # )
    # model.summary()