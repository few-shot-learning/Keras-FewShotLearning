from functools import wraps

from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import Conv2D, LeakyReLU, ZeroPadding2D, Add
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.regularizers import l2


@wraps(Conv2D)
def conv_2d(*args, **kwargs):
    return Conv2D(
        *args, **kwargs,
        kernel_regularizer=l2(5e-4),
        padding='valid' if kwargs.get('strides') == (2, 2) else 'same',
    )


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


def Darknet53(input_shape, *args, **kwargs):
    x = Input(input_shape)
    y = conv_block(32, (3, 3))(x)  # depth 4
    y = residual_block(y.shape[1:], 64, 1)(y)  # depth 15
    y = residual_block(y.shape[1:], 128, 2)(y)  # depth 33
    y = residual_block(y.shape[1:], 256, 8)(y)  # depth 93
    y = residual_block(y.shape[1:], 512, 8)(y)  # depth 153
    y = residual_block(y.shape[1:], 1024, 4)(y)  # depth 185
    return Model(x, y, *args, **kwargs)
