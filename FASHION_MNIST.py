import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
MNIST_model = keras.models.load_model('custom.h5')
img = tf.keras.utils.load_img(
    './images/cc.png',
    target_size=(28, 28),
    color_mode='grayscale'
)
img_array = tf.keras.utils.img_to_array(img)

image = np.expand_dims(img_array, axis=-1)
image = np.expand_dims(img_array, axis=0)
predictions = MNIST_model.predict(image)
score = tf.nn.softmax(predictions[0])

class_index = np.argmax(predictions[0])
class_names = ['bra', 'pantie', 'pants', 'short pants', 'skirt', 'socks']

print(class_index)
print('Predicted class:', class_names[class_index])

plt.imshow(img)
plt.show()
