from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


def conv_2d(*args, **kwargs):
    return Conv2D(
        *args,
        **kwargs,
        kernel_initializer=RandomNormal(0.0, 0.01),
        bias_initializer=RandomNormal(0.5, 0.01),
        kernel_regularizer=l2(),
    )


def KochNet(input_shape=(105, 105, 3)):
    """
    The conv net used as backbone in
    [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
    """

    model = Sequential(name="koch_net")
    model.add(Input(input_shape))
    model.add(conv_2d(64, (10, 10)))
    model.add(MaxPooling2D())
    model.add(conv_2d(128, (7, 7), activation="relu"))
    model.add(MaxPooling2D())
    model.add(conv_2d(128, (4, 4), activation="relu"))
    model.add(MaxPooling2D())
    model.add(conv_2d(256, (4, 4), activation="relu"))
    model.add(Flatten())
    model.add(
        Dense(4096, activation="sigmoid", kernel_initializer=RandomNormal(0.0, 0.2), bias_initializer=RandomNormal(0.5, 0.01))
    )

    return model
