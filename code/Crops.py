import tensorflow as tf
import keras
from tensorflow.keras import models, Sequential, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import shutil
from numpy import ceil
import matplotlib.pyplot as plt
import glob
base_path = 'F:\datasets\Primary Dataset'
test = os.path.join(base_path, 'test')
train = os.path.join(base_path, 'train')
train_smut = os.path.join(train, 'Smut')
train_blast = os.path.join(train, 'Blast')
train_Health = os.path.join(train, 'Healthy')
test_smut = os.path.join(test, 'Smut')
test_blast = os.path.join(test, 'Blast')
test_Health = os.path.join(test, 'Healthy')
train_smut_num = len(os.listdir(train_smut))
train_Health_num = len(os.listdir(train_Health))
train_Blast_num = len(os.listdir(train_blast))
test_smut_num = len(os.listdir(test_smut))
test_Health_num = len(os.listdir(test_Health))
test_Blast_num = len(os.listdir(test_blast))
total_test_num = test_Blast_num + test_Health_num + test_smut_num
total_train_num = train_Blast_num + train_Health_num + train_smut_num

print(total_test_num)
print(total_train_num)
img_size = 150
train_image_generator = ImageDataGenerator(rescale=1. / 255.0,
                                           rotation_range=0.2,
                                           vertical_flip=True,
                                           zoom_range=0.2,
                                           shear_range=0.2,
                                           height_shift_range=0.2,
                                           fill_mode='nearest')
test_image_generator = ImageDataGenerator(rescale=1. / 255.0)
train_data_gen = train_image_generator.flow_from_directory(
    train,
    batch_size=32,
    shuffle=True,
    target_size=(img_size, img_size),
    class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(
    directory=test,
    shuffle=True,
    batch_size=32,
    target_size=(img_size, img_size),
    class_mode='binary')
shape = img_size, img_size, 3
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(shape)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_data_gen,
                    steps_per_epoch=int(np.ceil(total_train_num / 32.0)),
                    epochs=50,
                    validation_steps=int(np.ceil(total_test_num / 32.0)),
                    validation_data=test_data_gen)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(50)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')