import tensorflow as tf
from tensorflow import keras


class image_net(keras.Model):
    def __init__(self, image_dims=(10,10,3)):
        super(image_net, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=6, kernel_size=8, strides=3, padding='valid', activation='relu')
