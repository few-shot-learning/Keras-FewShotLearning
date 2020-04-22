"""
All these metrics functions assume y_pred is a gram matrix computed on the batch (output of GramMatrix layer for
instance). y_true should be one-hot encoded
"""
import tensorflow as tf


def top_score_classification_accuracy(y_true, y_pred):
    """
    Use the top score of a sample against all the other sample to get a predicted label for each sample.
    Then mean accuracy is returned.

    Note: if there is no other sample of the same class, the sample will always be counted as failure
    since it is not possible to find the right class in the other samples.
    """
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_pred = y_pred - tf.linalg.diag(tf.linalg.diag_part(y_pred))
    y_pred = tf.map_fn(lambda x: y_true[x], tf.math.argmax(y_pred, axis=1), dtype=y_pred.dtype)
    return tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))


def mean_score_classification_accuracy(y_true, y_pred):
    """
    Use the mean score of an image against all the samples from the same class to get a score per class for each image.
    """
    y_pred = tf.linalg.normalize(
        tf.linalg.matmul(y_pred, tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=0))), ord=1, axis=1
    )[0]
    y_pred = tf.map_fn(lambda x: y_true[x], tf.math.argmax(y_pred, axis=1), dtype=y_pred.dtype)
    return tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))


def same_image_score(_, y_pred):
    """
    Same image score may not be always zero especially when bias is used in the head_model
    """
    return tf.reduce_mean(tf.linalg.diag_part(y_pred))


def accuracy(margin=0.0):
    """
    Compute the relative number of pairs with a score in the margin, ie. #{pairs | |y_true - y_pred| < m}
    """

    def _accuracy(y_true, y_pred):
        y_true = tf.matmul(y_true, y_true, transpose_b=True)
        return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) < margin, y_pred.dtype))

    return _accuracy


def min_eigenvalue(_, y_pred):
    """
    Compute the minimum eigenvalue of the y_pred tensor. If this value if non-negative (resp. positive) then the
    similarity or distance learnt is a positive semi-definite (resp. positive definite) kernel.
    See Also [Positive-definite kernel](https://en.wikipedia.org/wiki/Positive-definite_kernel)
    """
    return tf.reduce_min(tf.linalg.svd(y_pred, compute_uv=False))
