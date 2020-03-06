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
    return K.binary_crossentropy(
        y_true, tf.math.divide_no_nan(tf.linalg.matmul(y_pred, y_true), tf.reduce_sum(y_true, axis=0)),
    )


def binary_crossentropy(margin=0.0):
    """
    Compute the binary crossentropy loss of each possible pair in the batch. The margin lets define a threshold against
    which the difference is not taken into account, ie. |y_true - y_pred| < margin => loss = 0
    """

    def _binary_crossentropy(y_true, y_pred):
        y_true = tf.matmul(y_true, y_true, transpose_b=True)
        return (K.cast(K.abs(y_true - y_pred) > margin, "float32")) * K.binary_crossentropy(y_true, y_pred)

    return _binary_crossentropy


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
