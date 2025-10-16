import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
import os


def make_model(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, train_size=0.8,random_state=42,stratify=y_data)
    training = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32)
    testing = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation = "relu", input_shape = (128,180,1)))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64,(3,3), activation = "relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.Conv2D(128, (3,3), activation = "relu"))
    model.add(layers.MaxPool2D(2,2))

    model.add(layers.GlobalAveragePooling2D(keepdims = False))

    model.add(layers.Dense(units = 64,activation = "relu"))
    model.add(layers.Dense(units = 6, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
    
    model.fit(training, epochs = 20, validation_data = testing)
    os.makedirs("saved_models",exist_ok=True)
    model.save("saved_models/emotion_model_ver_one.h5")
    return model, X_train, y_train, X_test, y_test, testing


