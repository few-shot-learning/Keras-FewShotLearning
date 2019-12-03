import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class Classification(Layer):
    """
    Uses the inner kernel to compute the score matrix between batch and support set, eventually returns
    the average score per class
    """

    def __init__(self, kernel, support_tensors, support_labels, **kwargs):
        """
        Args:
            support_tensors (tf.Tensor): support set embeddings with shape (n, *embedding_shape)
            support_labels (tf.Tensor): one-hot encoded support set labels with shape (n, n classes)
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.support_tensors = tf.Variable(
            [], validate_shape=False, shape=tf.TensorShape((None, None)), name='support_tensors',
        )
        self.support_labels = tf.Variable(
            [], validate_shape=False, shape=tf.TensorShape((None, None)), name='support_labels',
        )
        self.set_support_set(support_tensors, support_labels)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel,
            'support_tensors': self.support_tensors,
            'support_labels': self.support_labels,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def _validate_support_set_shape(support_tensors, support_labels):
        if support_tensors.shape[0] != support_labels.shape[0]:
            raise AttributeError('Support tensors and support labels shape 0 should match')
        if support_tensors.shape[0] == 0:
            raise AttributeError('Support set cannot be empty')

    def set_support_set(self, support_tensors, support_labels):
        self._validate_support_set_shape(support_tensors, support_labels)
        K.batch_set_value([(self.support_tensors, support_tensors), (self.support_labels, support_labels)])

    def compute_output_shape(self, input_shape):
        return input_shape[0], tf.shape(self.support_labels)[1]

    @tf.function
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                raise AttributeError('Layer should be called on a single tensor')
            inputs = inputs[0]
        batch_size = tf.shape(inputs)[0]
        support_set_size = tf.shape(self.support_tensors)[0]
        pair_wise_scores = tf.reshape(
            self.kernel([
                tf.reshape(tf.tile(inputs, [1, support_set_size]), [-1, inputs.shape[1]]),
                tf.tile(self.support_tensors, [batch_size, 1]),
            ]),
            [batch_size, support_set_size],
        )
        return (
            tf.math.divide_no_nan(
                tf.matmul(pair_wise_scores, self.support_labels),
                tf.reduce_sum(self.support_labels, axis=0),
            )
        )
