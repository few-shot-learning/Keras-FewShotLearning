from functools import wraps

from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, ZeroPadding2D, Add, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2


@wraps(Conv2D)
def conv_2d(*args, **kwargs):
    return Conv2D(*args, **kwargs, kernel_regularizer=l2(5e-4), padding="valid" if kwargs.get("strides") == (2, 2) else "same")


def conv_block(*args, **kwargs):
    layer = Sequential()
    layer.add(conv_2d(*args, **kwargs, use_bias=False))
    layer.add(BatchNormalization())
    layer.add(LeakyReLU(alpha=0.1))
    return layer


def residual_block(input_shape, num_filters, num_blocks):
    x = Input(input_shape)
    y = ZeroPadding2D(((1, 0), (1, 0)))(x)
    y = conv_block(num_filters, (3, 3), strides=(2, 2))(y)
    for _ in range(num_blocks):
        residual = y
        y = conv_block(num_filters // 2, (1, 1))(y)
        y = conv_block(num_filters, (3, 3))(y)
        y = Add()([residual, y])
    return Model(x, y)


def Darknet53(input_shape=(224, 224, 3), *args, **kwargs):
    x = Input(input_shape)
    y = conv_block(32, (3, 3))(x)  # 1
    y = residual_block(y.shape[1:], 64, 1)(y)  # 4
    y = residual_block(y.shape[1:], 128, 2)(y)  # 9
    y = residual_block(y.shape[1:], 256, 8)(y)  # 26
    y = residual_block(y.shape[1:], 512, 8)(y)  # 43
    y = residual_block(y.shape[1:], 1024, 4)(y)  # 52
    return Model(x, y, *args, **kwargs)


def Darknet7(input_shape=(224, 224, 3), *args, **kwargs):
    """
    Backbone of tiny YOLO
    """
    return Sequential(
        [
            conv_block(16, (3, 3), input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            conv_block(32, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            conv_block(64, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            conv_block(128, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            conv_block(256, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            conv_block(512, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            conv_block(1024, (3, 3)),
        ],
        *args,
        **kwargs,
    )
