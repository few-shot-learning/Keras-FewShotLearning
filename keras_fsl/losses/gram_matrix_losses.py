"""
Base class for all losses to be applied on a Gram matrix like output, ie when the output y_pred of the network is the the pair-wise
distance / similarity of all items of the batch (see GramMatrix layer for instance). y_true should be one-hot encoded.

For unsupervised metric learning, it is standard to use each image instance as a distinct class of its own. In this settings all
the losses are directly available by setting label = image_id and y_true stands indeed for all the patches/glimpses/etc. extracted from
the same image. It is usually supposed that the risk of collision is low. For more information on unsupervised learning of visual
representation, see for instance
[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
[Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
[Unsupervised feature learning via non-parametric instance discrimination](https://arxiv.org/abs/1805.01978v1)
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class MeanScoreClassificationLoss(Loss):
    """
    Use the mean score of an image against all the samples from the same class to get a score per class for each image.
    """

    def call(self, y_true, y_pred):
        y_pred = tf.linalg.normalize(y_pred @ tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=0)), ord=1, axis=1)[0]
        return tf.reduce_sum(K.binary_crossentropy(y_true, y_pred) * y_true, axis=1)


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
    return K.binary_crossentropy(identity_matrix, confusion_matrix)


class ClassConsistencyLoss(Loss):
    def call(self, y_true, y_pred):
        return class_consistency_loss(y_true, y_pred)


class BinaryCrossentropy(Loss):
    """
    Compute the binary crossentropy loss of each possible pair in the batch.
    The margins lets define a threshold against which the difference is not taken into account,
    ie. only values with lower < |y_true - y_pred| < upper will be non-zero

    Args:
        lower (float): ignore loss values below this threshold. Useful to make the network focus on more significant errors
        upper (float): ignore loss values above this threshold. Useful to prevent the network from focusing on errors due to
            wrongs labels (or collision in unsupervised learning)
    """

    def __init__(self, lower=0.0, upper=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, y_true, y_pred):
        adjacency_matrix = tf.matmul(y_true, y_true, transpose_b=True)
        clip_mask = tf.math.logical_and(
            tf.abs(adjacency_matrix - y_pred) < self.upper, tf.abs(adjacency_matrix - y_pred) > self.lower
        )
        return tf.cast(clip_mask, dtype=y_pred.dtype) * K.binary_crossentropy(adjacency_matrix, y_pred)


def max_crossentropy(y_true, y_pred):
    # TODO: use reduction kwarg of loss instead when possible
    return tf.reduce_max(BinaryCrossentropy()(y_true, y_pred))


def std_crossentropy(y_true, y_pred):
    # TODO: use reduction kwarg of loss instead when possible
    return tf.math.reduce_std(BinaryCrossentropy()(y_true, y_pred))


class TripletLoss(Loss):
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

    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
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
        loss_negatives_outside = tf.maximum(negatives_outside + tf.squeeze(triplets_min) - K.epsilon() + self.margin, 0)

        # 2) negatives_inside: greatest negative distance smaller than positive one
        loss_negatives_inside = tf.maximum(
            tf.reduce_min((triplets - triplets_max + K.epsilon()) * (1 - farther_negative_mask) * anp_mask, axis=-1)
            + tf.squeeze(triplets_max)
            - K.epsilon()
            + self.margin,
            0,
        )

        all_losses = tf.where(negatives_outside_mask, loss_negatives_outside, loss_negatives_inside)
        true_triplets_mask = adjacency_matrix - tf.eye(tf.shape(y_true)[0])
        return tf.reduce_sum(all_losses * true_triplets_mask) / tf.reduce_sum(true_triplets_mask)
