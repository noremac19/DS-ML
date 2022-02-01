from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Concatenate, BatchNormalization, Add
from keras.layers import Conv2D, MaxPooling2D, ReLU
from keras import backend as K
import tensorflow as tf
batch_size = 64
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols, channels = 28, 28, 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

############################# Architecture made by Ennui
x0 = Input(shape=input_shape)
x25 = Conv2D(16, (3,3),strides=(1,1), activation='relu', padding='same')(x0)
x29 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x25)
x28 = Conv2D(16, (3,3),strides=(1,1), padding='same')(x29)
x30 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x28)
x31 = Dropout(rate=0.5)(x30)
x26 = Flatten()(x31)
x27 = Dense(32, activation='relu')(x26)
x1 = Dense(10, activation='softmax')(x27)
model = Model(inputs=x0, outputs=x1)
#############################

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
