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

label = ['bra', 'pantie', 'pants', 'shirt',
               'short pants', 'skirt', 'socks', 't-shirt']

idx = 0
class_label = dict()
for i in label:
  class_label[i] = idx
  idx += 1
class_label

plt.figure(figsize=(10, 10))
for images, labels in train_set.take(1):
  for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_set.class_names[labels[i]])
    plt.axis("off")

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()