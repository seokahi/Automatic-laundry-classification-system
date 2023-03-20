import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 사이즈를 28x28로 조정합니다.
img_height = 28
img_width = 28

# 데이터를 불러옵니다.
train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'underwear',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)

# 모델을 생성합니다.
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델을 컴파일합니다.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델을 학습합니다.
model.fit(train_data, epochs=10)
