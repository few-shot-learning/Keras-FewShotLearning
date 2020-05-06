from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D


def SingleConv2D(input_shape):
    """
    A Single Conv2D followed by a global average Pooling, mainly for debugging/testing purpose
    """
    model = Sequential()
    model.add(Conv2D(10, (3, 3), input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    return model
