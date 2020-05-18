import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf

def image_dir_loader(image_dir):
    #this is supposed to a list with all the images as tensors
    os.chdir(image_dir)
    images_batch = list()
    for image in os.listdir(image_dir):
        try:
            
            image_tensor = tf.convert_to_tensor(np.asarray(Image.open(image)))
            print(image_tensor.shape)
            images_batch.append(image_tensor)
        except:
            print('failed with file'+ image)
            continue
    return images_batch
    
directory = 'C:\\Users\\jesus\\Documents\\WorkFiles\\PythonCode\\TF2_personal_projects\\images_folder\\Saved Pictures'
image_dir_loader(directory)

class image_batch_loader:
    def __init__(self, batch_num, x_image_dir,y_image_dir):
        self.batch = batch_num
        self.x_image_dir = x_image_dir
        self.y_image_dir = y_image_dir
        self.x_images = image_dir_loader(self.x_image_dir)
        self.y_images = image_dir_loader(self.y_image_dir)

testing = True
