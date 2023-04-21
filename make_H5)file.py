import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
train_set = tf.keras.utils.image_dataset_from_directory(directory='/content/drive/MyDrive/cap/custom_dataset/',
                                        # rescale=1./255,
                                        image_size=(28,28),
                                        # image_size=(256,256),
                                        validation_split=0.2,
                                        seed=123,
                                        # class_mode='categorical',
                                        subset='training',
                                        color_mode='grayscale',
                                        batch_size=32)
valid_set = tf.keras.utils.image_dataset_from_directory(directory='/content/drive/MyDrive/cap/custom_dataset/',
                                        # rescale=1./255,
                                        image_size=(28,28),
                                        # image_size=(256,256),
                                        validation_split=0.2,
                                        seed=123,
                                        # class_mode='categorical',
                                        subset='validation',
                                        color_mode='grayscale',
                                        shuffle= False,
                                        batch_size=32)
image_data = tf.keras.utils.image_dataset_from_directory(directory='/content/drive/MyDrive/cap/custom_dataset/',
                                         image_size=(28,28))