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
    y_pred_by_label = tf.linalg.normalize(
        tf.linalg.matmul(y_pred, tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=0))), ord=1, axis=1
    )[0]
    return K.categorical_crossentropy(y_true, y_pred_by_label)


def class_consistency_loss(y_true, y_pred):
    """
    Use the mean score of an image against all the samples from the same class to get a score per class for each image.
    Then average again over all the samples to get a class_wise confusion matrix
    """
    y_true = tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=0))
    class_mask = tf.reduce_sum(y_true, axis=0) > 0
    confusion_matrix = tf.boolean_mask(
        tf.matmul(y_true, tf.matmul(y_pred, y_true), transpose_a=True)[class_mask], class_mask, axis=1
    )
    identity_matrix = tf.eye(tf.shape(confusion_matrix)[0])
    return tf.reduce_mean(K.binary_crossentropy(identity_matrix, confusion_matrix))


def binary_crossentropy(lower_margin=0.0, upper_margin=1.0):
    """
    Compute the binary crossentropy loss of each possible pair in the batch.
    The margins lets define a threshold against which the difference is not taken into account,
    ie. only values with lower_margin < |y_true - y_pred| < upper_margin will be non-zero

    Args:
        lower_margin (float): ignore errors below this threshold. Useful to make the network focus on more significant errors
        upper_margin (float): ignore errors above this threshold. Useful to prevent the network from focusing on errors due to
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


def triplet_loss(margin=1.0):
    """
    Implement triplet loss with semi-hard negative mining as in
    [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf):
    a triplet (A, N, P) consists of three images with two labels such that A and P are of the same class (positive pair) and A and N are
    from two different classes (negative pair).

    Then the loss tries to enforce the following relation: d(A, P) + margin < d(A,N) with d a given _distance_ function. In the original
    implementation as well as in the standard tf.addons one this is the euclidean distance. Here this can be any _kernel_ (see
    SupportLayer).

    Semi-hard mining means that given a batch of embeddings, all item in the batch are used as anchor and for all anchor, all positive pairs
    are used in a triplet and that the negative pair is chosen such that:
    1) it is the closest negative sample farther than the positive one
    2) or it is the farthest negative sample
    It means that we somehow select the negative sample the closest to the margin, but give a preference to sample beyond the margin.

    Args:
        margin (float): margin for separating positive to negative pairs

    """

    def _triplet_loss(y_true, y_pred):
        # 0) build triplets tensor such that triplet[a, p, n] = d(a, p) - d(a, n)
        adjacency_matrix = tf.matmul(y_true, y_true, transpose_b=True)
        anp_mask = tf.cast(tf.expand_dims(adjacency_matrix, -1) + tf.expand_dims(adjacency_matrix, 1) == 1, tf.float32)
        triplets = tf.expand_dims(y_pred, -1) - tf.expand_dims(y_pred, 1)

        triplets_max = tf.reduce_max(triplets, axis=-1, keepdims=True)
        triplets_min = tf.reduce_min(triplets, axis=-1, keepdims=True)
        farther_negative_mask = tf.cast(triplets < 0, tf.float32)

        # 1) negatives_outside: smallest negative distance greater than positive one
        negatives_outside = tf.reduce_max((triplets - triplets_min + K.epsilon()) * farther_negative_mask * anp_mask, axis=-1)
        negatives_outside_mask = negatives_outside > 0
        loss_negatives_outside = tf.maximum(negatives_outside + tf.squeeze(triplets_min) - K.epsilon() + margin, 0)

        # 2) negatives_inside: greatest negative distance smaller than positive one
        loss_negatives_inside = tf.maximum(
            tf.reduce_min((triplets - triplets_max + K.epsilon()) * (1 - farther_negative_mask) * anp_mask, axis=-1)
            + tf.squeeze(triplets_max)
            - K.epsilon()
            + margin,
            0,
        )

        all_losses = tf.where(negatives_outside_mask, loss_negatives_outside, loss_negatives_inside)
        true_triplets_mask = adjacency_matrix - tf.eye(tf.shape(y_true)[0])
        return tf.reduce_sum(all_losses * true_triplets_mask) / tf.reduce_sum(true_triplets_mask)

    return _triplet_loss
