import tensorflow as tf
import keras

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128,128,3)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(8, activation="softmax"))

train_ds = keras.utils.image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=8,
    image_size=(128, 128))
validation_ds = keras.utils.image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=8,
    image_size=(128, 128))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=validation_ds)
model.save("breed_classifier.h5")