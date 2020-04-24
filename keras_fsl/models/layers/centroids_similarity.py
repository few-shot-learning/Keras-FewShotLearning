import tensorflow as tf
from tensorflow.keras import activations

from keras_fsl.models.layers.support_layer import SupportLayer


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

    @tf.function
    def set_support_set(self, *args, **kwargs):
        super().set_support_set(*args, **kwargs)
        if tf.shape(self.support_labels)[1] == 1:
            columns, codes = tf.unique(self.support_labels)
            support_labels_one_hot = tf.one_hot(codes, depth=tf.size(columns))
            self.support_labels = columns
        else:
            support_labels_one_hot = self.support_labels
            self.support_labels = tf.range(tf.shape(self.support_labels)[1])

        self.support_tensors = tf.matmul(
            tf.math.divide_no_nan(support_labels_one_hot, tf.reduce_sum(support_labels_one_hot, axis=0)),
            self.support_tensors,
            transpose_a=True,
        )

    def call(self, *args, **kwargs):
        return self.activation(super().call(*args, **kwargs))
