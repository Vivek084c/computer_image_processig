import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers, models

def fcn8(input_shape, num_classes=18):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    pool1 = x

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    pool2 = x

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    pool3 = x

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    pool4 = x

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    pool5 = x

    # Convolutional layers to replace FC layers
    x = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(pool5)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)

    # Classifying layer
    x = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(x)
    score_fr = x

    # Deconvolution layers
    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same')(score_fr)
    score_pool4 = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(pool4)
    x = layers.Add()([x, score_pool4])

    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    score_pool3 = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(pool3)
    x = layers.Add()([x, score_pool3])

    # Final upsampling
    x = layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)

    # Output layer
    outputs = layers.Activation('softmax')(x)

    model = models.Model(inputs, outputs)

    return model

# Create the FCN-8s model
# fcn8_model = fcn8(input_shape=(640, 832, 3), num_classes=18)
# fcn8_model.summary()
