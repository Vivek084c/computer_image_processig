import tensorflow as tf
from tensorflow.keras import layers, models

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

# Create the model
model = fcn16(input_shape=(512, 512, 3), num_classes=18)
model.summary()
