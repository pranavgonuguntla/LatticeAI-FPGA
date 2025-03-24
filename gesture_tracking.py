import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GestureCNN(keras.Model):
    def __init__(self, num_classes=6): # Num_classes depends on the number of different gestures in the dataset (6 is placeholder)
        super(GestureCNN, self).__init__()

        # Convolutional layers
        self.conv_one = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", input_shape=(64, 64, 1))
        self.conv_two = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")
        self.conv_three = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")