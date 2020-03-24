"""
All these loss functions assume y_pred is a gram matrix computed on the batch (output of GramMatrix layer for
instance). y_true should be one-hot encoded
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def mean_score_classification_loss(y_true, y_pred):
    """
    Use the mean score of an image against all the samples from the same class to get a score per class for each image.
    """
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        tf.argmax(y_true, axis=1),
        tf.math.divide_no_nan(tf.linalg.matmul(y_pred, tf.dtypes.cast(y_true, y_pred.dtype)), tf.reduce_sum(y_true, axis=0)),
    )


def class_consistency_loss(y_true, y_pred):
    """
    Use the mean score of an image against all the samples from the same class to get a score per class for each image. Then average again
    over all the samples to get a class_wise confusion matrix
    """
    y_true = tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=0))
    confusion_matrix = tf.matmul(y_true, tf.matmul(y_pred, y_true), transpose_a=True)
    identity_matrix = tf.eye(tf.shape(y_true)[1])
    return tf.reduce_mean(K.binary_crossentropy(identity_matrix, confusion_matrix))


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
        return tf.cast(keep_loss, dtype=y_pred.dtype) * K.binary_crossentropy(y_true, y_pred)

    return _binary_crossentropy


def max_crossentropy(y_true, y_pred):
    return tf.reduce_max(binary_crossentropy()(y_true, y_pred))


def std_crossentropy(y_true, y_pred):
    return tf.math.reduce_std(binary_crossentropy()(y_true, y_pred))
