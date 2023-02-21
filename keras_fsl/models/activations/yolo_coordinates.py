"""
Activation function for mapping feature into output coordinates as in Yolo V3
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda


@tf.function
def build_grid_coordinates(grid_shape):
    """
    Build a grid coordinate tensor with shape (*grid_shape, 2) where grid[i, j, 0] = i and grid[i, j, 1] = j
    Args:
        grid_shape (Union[tuple, list, tensorflow.TensorShape]): to be passed to tf.range

    Returns:
        (tensorflow.Tensor)
    """
    height, width = tf.meshgrid(tf.range(0, grid_shape[0]), tf.range(0, grid_shape[1]))
    width = tf.transpose(width)
    height = tf.transpose(height)
    return tf.stack([height, width], -1)


def YoloCoordinates():
    """
    Activation function for the box center coordinates regression. Coordinates are relative to the image dimension, ie. between 0
        and 1
    """
    return Sequential(
        [
            Activation("sigmoid"),
            Lambda(
                lambda input_: input_ + tf.cast(tf.expand_dims(build_grid_coordinates(tf.shape(input_)[1:3]), 0), input_.dtype)
            ),
            Lambda(lambda input_: input_ / tf.cast(tf.shape(input_)[1:3], input_.dtype)),
        ]
    )
