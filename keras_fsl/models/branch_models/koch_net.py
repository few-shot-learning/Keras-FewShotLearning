from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


def KochNet(input_shape=(105, 105, 3)):
    """
    The conv net used as backbone in
    [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
    """

    model = Sequential(name='koch_net')
    model.add(Input(input_shape))
    model.add(Conv2D(64, (10, 10)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    return model
