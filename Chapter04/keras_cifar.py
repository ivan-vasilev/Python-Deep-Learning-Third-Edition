print("CIFAR-10 with Keras convolutional network")

import keras
from keras.datasets import cifar10

(X_train, Y_train), (X_validation, Y_validation) = cifar10.load_data()

Y_train = keras.utils.to_categorical(Y_train, 10)
Y_validation = keras.utils.to_categorical(Y_validation, 10)

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)

# Apply z-normalization on the training set
data_generator.fit(X_train)

# Standardize the validation set
X_validation = data_generator.standardize(X_validation.astype('float32'))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization

model = Sequential(layers=[
    Conv2D(32, (3, 3),
           padding='same',
           input_shape=X_train.shape[1:]),
    BatchNormalization(),
    Activation('gelu'),
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('gelu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('gelu'),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('gelu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3)),
    BatchNormalization(),
    Activation('gelu'),
    Conv2D(128, (3, 3)),
    BatchNormalization(),
    Activation('gelu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 50

model.fit(
    x=data_generator.flow(x=X_train,
                          y=Y_train,
                          batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=100,
    verbose=1,
    validation_data=(X_validation, Y_validation),
    workers=4)
