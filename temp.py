import numpy as np
import cv2
# Shape of the array (height, width, channels)
shape = (128, 128, 3)

# Create a numpy array with all elements set to 0 (black color)
black_image = np.zeros((128, 128, 3), dtype=np.uint8)

cv2.imwrite("black image.jpg" ,black_image)