import tensorflow as tf
from tensorflow.keras import activations

from keras_fsl.layers.support_layer import SupportLayer


class CentroidsMatrix(SupportLayer):
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

    def build_support_set(self, inputs):
        """
        Args:
            inputs (List[tf.Tensor]): should be [embeddings, labels] in this order. Labels are assumed to be one-hot encoded
        """
        if not isinstance(inputs, list) or isinstance(inputs, list) and len(inputs) != 2:
            raise ValueError(f"{self.__class__.__name__} should be called on a list of inputs [embeddings, labels_one_hot]")
        embeddings = inputs[0]
        labels_one_hot_normalized = tf.math.divide_no_nan(inputs[1], tf.reduce_sum(inputs[1], axis=0))
        return tf.matmul(labels_one_hot_normalized, embeddings, transpose_a=True)

    def call(self, *args, **kwargs):
        return self.activation(super().call(*args, **kwargs))
