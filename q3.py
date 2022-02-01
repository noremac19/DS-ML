import keras
from keras.datasets import cifar100
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

for i in range(30):
    row = X_train.iloc[i, :]
    row = np.array(row)
    row = row.reshape(32, 32)
    plt.imshow(row)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

img_rows, img_cols, channels = 32, 32, 3

input_shape = (32,32,3)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))
#############################

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

model = model.fit(X_train, y_train,
          batch_size=50,
          epochs=5,
          verbose=1,
          validation_data=(X_test, y_test))


plt.figure(1)
plt.title("Train Loss")
plt.plot(model.history['loss'])
plt.ylabel("Loss")
plt.xlabel("epoch")

plt.figure(2)
plt.title("Validation Loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.plot(model.history['val_loss'])

plt.figure(3)
plt.title("Train Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.plot(model.history['accuracy'])

plt.figure(4)
plt.title("Validation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.plot(model.history['val_accuracy'])
plt.show()


