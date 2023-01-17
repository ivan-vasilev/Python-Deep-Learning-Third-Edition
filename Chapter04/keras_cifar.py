print("CIFAR-10 with Keras convolutional network")

from tensorflow import keras

(X_train, Y_train), (X_validation, Y_validation) = \
    keras.datasets.cifar10.load_data()

Y_train = keras.utils.to_categorical(Y_train, 10)
Y_validation = keras.utils.to_categorical(Y_validation, 10)

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True)

# Apply z-normalization on the training set
data_generator.fit(X_train)

# Standardize the validation set
X_validation = data_generator.standardize(X_validation.astype('float32'))

from keras import layers, models

model = models.Sequential(layers=[
    layers.Conv2D(32, (3, 3),
                  padding='same',
                  input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.Activation('gelu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('gelu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('gelu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('gelu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('gelu'),
    layers.Conv2D(128, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('gelu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(10, activation='softmax')
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
