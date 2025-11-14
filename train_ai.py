import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2

train_datagen = ImageDataGenerator(
    rescale=1./255,              
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2      
)

train_generator = train_datagen.flow_from_directory(
    '/Users/bogdy2k/Desktop/Projects/Ia-proiect/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'         
)

validation_generator = train_datagen.flow_from_directory(
    '/Users/bogdy2k/Desktop/Projects/Ia-proiect/dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'       
)

base_model = DenseNet121(weights = 'imagenet' , include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D() (x)
x = Dense(256 , activation= 'relu') (x)
x = Dropout(0.5) (x)
predictions = Dense(train_generator.num_classes , activation = 'softmax' ) (x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(
    optimizer='adam' , 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
callbacks = [checkpoint, earlystop, reduce_lr]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=callbacks
)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

plt.title('Loss')

plt.show()