from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, MaxPooling2D, Flatten


def conv_block(*args, **kwargs):
    layer = Sequential()
    layer.add(Conv2D(64, (3, 3), padding="same", *args, **kwargs))
    layer.add(BatchNormalization())
    layer.add(Activation("relu"))
    layer.add(MaxPooling2D((2, 2)))
    return layer


def VinyalsNet(input_shape=(28, 28, 3)):
    """
    The conv net used as backbone in
    [Matching networks for one shot learning](https://arxiv.org/abs/1606.04080) and in
    [Protypical networks](https://arxiv.org/pdf/1703.05175.pdf)
    """
    model = Sequential(name="vinyals_net")
    model.add(Input(input_shape))
    model.add(conv_block())
    model.add(conv_block())
    model.add(conv_block())
    model.add(conv_block())
    model.add(Flatten())
    return model
