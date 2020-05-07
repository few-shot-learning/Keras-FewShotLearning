from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def BasicCNN(input_shape, classes=None):
    model = Sequential(
        [
            Conv2D(32, (3, 3), input_shape=input_shape, padding="same", activation="relu"),
            Dropout(0.2),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
        ]
    )
    if classes is not None:
        model.add(Dense(classes, activation="softmax"))
    return model
