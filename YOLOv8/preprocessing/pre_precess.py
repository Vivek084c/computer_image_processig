import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
#creating the directories

class PreProcess:
    def __init__(self, directories, COLORMAP):
        self.directories = directories
        self.COLORMAP = COLORMAP
    
    def create_dir(self):
        """
        create necessary directory 
        Args:
            None
        return:
            None
        """
        for path in self.directories:
            if not os.path.exists(path):
                os.makedirs(path)

    def load_dataset(dataset_path, split=0.2, cap_size=300):
        """"
        splits the dataset into train, test, validation set
        Args:
            dataset_path: path to the root dataset path
            split: split size for validation and test dataset
            cap_size: how many files to read (by default 300 out of around 7000)
        return:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) -> returns the filepath for train, test and validation dataeset
        """
        train_x = sorted(glob(os.path.join(dataset_path, "input_1", "*")))[:cap_size]
        train_y = sorted(glob(os.path.join(dataset_path, "Categories_1", "*")))[:cap_size]

        split_size = int(split * len(train_x))

        train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
        train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

        train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    

    def read_image(self, x, IMG_W, IMG_H):
        """
        takes an input image path and returns the normalized rgb matrix
        Args:
            x: the file path
            IMG_W: width of the imgae vector
            IMG_H: height of the image vector 
        return:
            normalized rgb matrix of image
        """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (IMG_W, IMG_H))  # Resize to (IMG_W, IMG_H)
        x = x / 255.0
        x = x.astype(np.float32)
        return x

    def read_mask(self,x, IMG_W, IMG_H):
        """
        takes in the image path and returns the binary color map for all classes
        Args:
            x: the file path
            IMG_W: width of the imgae vector
            IMG_H: height of the image vector 
        return:
            binary mask stacked on top of each other with specific number of class (= the number of classes in segmetation model) 
        """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (IMG_W, IMG_H))  # Resize to (IMG_W, IMG_H)

        # Processing the mask
        output = []
        for color in self.COLORMAP:
            cmap = np.all(np.equal(x, color), axis=-1)
            output.append(cmap)
        output = np.stack(output, axis=-1)
        output = output.astype(np.uint8)
        return output
    