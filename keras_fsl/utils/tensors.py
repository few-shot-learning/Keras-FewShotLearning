import tensorflow as tf


def get_dummies(x):
    """
    Pendant to pandas.get_dummies method but for tf.Tensor. Tensor is flatten first.
    Args:
        x (tf.Tensor): input tensor with dtype supported by tf.unique

    Returns:
        tf.Tensor, tf.Tensor: the one-hot encoded input as well as the column names
    """
    columns, codes = tf.unique(tf.reshape(x, [-1]))
    return tf.one_hot(codes, depth=tf.size(columns)), columns
