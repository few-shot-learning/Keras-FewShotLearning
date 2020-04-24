import tensorflow as tf
from tensorflow.keras import activations

from keras_fsl.models.layers.support_layer import SupportLayer
from keras_fsl.utils.tensors import get_dummies


class CentroidsSimilarity(SupportLayer):
    """
    Compute the similarity (distance) between all items of the input batch and the centroid of each class found in the batch.
    """

    def __init__(self, kernel, activation="linear", **kwargs):
        """

        Args:
            kernel: similarity / distance function (x, x') => tf.float32
            activation: add an activation function as in other keras standard layers (e.g. softmax to compute ProtoNets output).
        """
        super().__init__(kernel, **kwargs)
        self.activation = activations.get(activation)

    def set_support_set(self, *args, **kwargs):
        super().set_support_set(*args, **kwargs)
        support_labels_one_hot = tf.cond(
            tf.shape(self.support_labels)[1] == 1, lambda: get_dummies(self.support_labels)[0], lambda: self.support_labels
        )
        self.support_tensors = tf.matmul(
            tf.math.divide_no_nan(support_labels_one_hot, tf.reduce_sum(support_labels_one_hot, axis=0)),
            self.support_tensors,
            transpose_a=True,
        )

    def call(self, *args, **kwargs):
        return self.activation(super().call(*args, **kwargs))
