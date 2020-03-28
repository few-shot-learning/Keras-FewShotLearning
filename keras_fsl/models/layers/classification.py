import tensorflow as tf
from tensorflow.keras.layers import Layer

from keras_fsl.losses import class_consistency_loss


class Classification(Layer):
    """
    Uses the inner kernel to compute the score matrix between batch and support set, eventually returns
    the average score per class
    """

    support_tensors_shape = tf.TensorShape([None, None])
    support_labels_shape = tf.TensorShape([None, None])
    support_tensors_spec = tf.TensorSpec(support_tensors_shape, tf.float32, name="support_tensors")
    support_labels_spec = tf.TensorSpec(support_labels_shape, tf.float32, name="support_labels")

    def __init__(self, kernel, **kwargs):
        """
        Args:
            support_tensors (tf.Tensor): support set embeddings with shape (n, *embedding_shape)
            support_labels (tf.Tensor): one-hot encoded support set labels with shape (n, n classes)
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.support_tensors = tf.Variable([[]], validate_shape=False, shape=self.support_tensors_shape, name="support_tensors")
        self.support_labels = tf.Variable([[]], validate_shape=False, shape=self.support_labels_shape, name="support_labels")
        self.support_set_loss = tf.Variable(0.0, name="support_set_loss")

    def get_config(self):
        config = super().get_config()
        config.update({"kernel": self.kernel.to_json()})
        return config

    @classmethod
    def from_config(cls, config):
        kernel = tf.keras.models.model_from_json(config["kernel"])
        config["kernel"] = kernel
        return cls(**config)

    @staticmethod
    def _validate_support_set_shape(support_tensors, support_labels):
        if support_tensors.shape[0] != support_labels.shape[0]:
            raise AttributeError("Support tensors and support labels shape 0 should match")

    @tf.function(input_signature=(support_tensors_spec, support_labels_spec, tf.TensorSpec(None, tf.bool, name="overwrite")))
    def set_support_set(self, support_tensors, support_labels, overwrite):
        self._validate_support_set_shape(support_tensors, support_labels)
        support_tensors = tf.cond(
            overwrite, lambda: support_tensors, lambda: tf.concat([self.support_tensors, support_tensors], axis=0)
        )
        support_labels = tf.cond(
            overwrite, lambda: support_labels, lambda: tf.concat([tf.math.ceil(self.support_labels), support_labels], axis=0)
        )
        support_set_size = tf.shape(support_tensors)[0]
        pair_wise_scores = tf.reshape(
            self.kernel(
                [
                    tf.repeat(support_tensors, tf.ones(support_set_size, dtype=tf.int32) * support_set_size, axis=0),
                    tf.tile(support_tensors, [support_set_size, 1]),
                ]
            ),
            [support_set_size, support_set_size],
        )
        self.support_set_loss.assign(class_consistency_loss(support_labels, pair_wise_scores))

        normalized_labels = tf.math.divide_no_nan(support_labels, tf.reduce_sum(support_labels, axis=0))
        self.support_labels.assign(normalized_labels)
        self.support_tensors.assign(support_tensors)
        return tf.expand_dims(self.support_set_loss, axis=0)

    @tf.function(input_signature=())
    def get_support_set(self):
        return self.support_tensors, self.support_labels, self.support_set_loss

    def compute_output_shape(self, input_shape):
        return input_shape[0], tf.shape(self.support_labels)[1]

    @tf.function
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                raise ValueError("Layer should be called on a single tensor")
            inputs = inputs[0]
        batch_size = tf.shape(inputs)[0]
        support_set_size = tf.shape(self.support_tensors)[0]
        pair_wise_scores = tf.reshape(
            self.kernel(
                [
                    tf.repeat(inputs, tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * support_set_size, axis=0),
                    tf.tile(self.support_tensors, [batch_size, 1]),
                ]
            ),
            [batch_size, support_set_size],
        )
        return tf.linalg.normalize(tf.linalg.matmul(pair_wise_scores, self.support_labels), ord=1, axis=1,)[0]
