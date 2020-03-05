"""
Activation function for mapping feature into output box dimensions as in Yolo V3
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda


def YoloBox(anchor):
    """
    Activation function for the box dimension regression. Dimensions are relative to the image dimension, ie. between 0
        and 1

    Args:
        anchor (Union[pandas.Series, collections.namedtuple]): with key width and height. Note that given a tensor with shape
            (batch_size, i, j, channels), i is related to height and j to width
  """
    return Sequential(
        [
            Activation("exponential"),
            Lambda(
                lambda input_, anchor_=anchor: (
                    input_ * tf.convert_to_tensor([anchor_.height, anchor_.width], dtype=tf.float32)
                )
            ),
        ]
    )
