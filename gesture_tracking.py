import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GestureCNN(keras.Model):
    def __init__(self, num_classes=6): # Num_classes depends on the number of different gestures in the dataset (6 is placeholder)
        super(GestureCNN, self).__init__()

        # Convolutional layers and pooling layers
        self.conv_one = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", input_shape=(64, 64, 1))
        self.pool_one = layers.MaxPooling2D(pool_size=(2,2), strides=2)

        self.conv_two = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")
        self.pool_two = layers.MaxPooling2D(pool_size=(2,2), strides=2)

        self.conv_three = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")
        self.pool_three = layers.MaxPooling2D(pool_size=(2,2), strides=2)

        