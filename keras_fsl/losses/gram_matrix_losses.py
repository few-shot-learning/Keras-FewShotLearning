"""
All these loss functions assume y_pred is a gram matrix computed on the batch (output of GramMatrix layer for
instance). y_true should be one-hot encoded
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def mean_score_classification_loss(y_true, y_pred):
    """
    Use the mean score of an image against all the sample from the same class to get a score per class for each image.
    Then softmax is applied to use a classical categorical_crossentropy loss.
    """
    y_true = tf.dtypes.cast(y_true, tf.float32)
    return K.binary_crossentropy(y_true, tf.math.divide_no_nan(tf.linalg.matmul(y_pred, y_true), tf.reduce_sum(y_true, axis=0)))


def binary_crossentropy(lower_margin=0.0, upper_margin=1.0):
    """
    Compute the binary crossentropy loss of each possible pair in the batch. The margins lets define a threshold against
    which the difference is not taken into account, ie. only values with lower_margin < |y_true - y_pred| < upper_margin will be non-zero

    Args:
        lower_margin (float): ignore errors below this threshold. This can be useful to make the network focus on more significant errors
        upper_margin (float): ignore errors above this threshold. This can be useful to prevent the network from focusing on errors due to
            wrongs labels
    """

    def _binary_crossentropy(y_true, y_pred):
        y_true = tf.matmul(y_true, y_true, transpose_b=True)
        keep_loss = tf.math.logical_and(tf.abs(y_true - y_pred) < upper_margin, tf.abs(y_true - y_pred) > lower_margin)
        return tf.cast(keep_loss, dtype=tf.float32) * K.binary_crossentropy(y_true, y_pred)

    return _binary_crossentropy


def max_crossentropy(y_true_, y_pred):
    return tf.reduce_max(binary_crossentropy(0.0)(y_true_, y_pred))


def std_crossentropy(y_true_, y_pred):
    return tf.math.reduce_std(binary_crossentropy(0.0)(y_true_, y_pred))


def accuracy(margin=0.0):
    """
    Compute the relative number of pairs with a score in the margin, ie. #{pairs | |y_true - y_pred| < m}
    """

    def _accuracy(y_true, y_pred):
        y_true = tf.matmul(y_true, y_true, transpose_b=True)
        return K.mean(K.cast(K.abs(y_true - y_pred) < margin, "float32"))

    return _accuracy


def min_eigenvalue(_, y_pred):
    """
    Compute the minimum eigenvalue of the y_pred tensor. If this value if non-negative (resp. positive) then the
    similarity or distance learnt is a positive semi-definite (resp. positive definite) kernel.
    See Also [Positive-definite kernel](https://en.wikipedia.org/wiki/Positive-definite_kernel)
    """
    return tf.reduce_min(tf.linalg.svd(y_pred, compute_uv=False))
