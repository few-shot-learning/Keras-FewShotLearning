import abc
import tensorflow as tf
from tensorflow.keras.layers import Layer


class SupportLayer(Layer, metaclass=abc.ABCMeta):
    """
    An abstract layer that stores internally a support set for inference
    """

    support_tensors_shape = tf.TensorShape([None, None])
    support_labels_one_hot_shape = tf.TensorShape([None, None])
    support_labels_name_shape = tf.TensorShape([None])
    support_tensors_spec = tf.TensorSpec(support_tensors_shape, tf.float32, name="support_tensors")
    support_labels_one_hot_spec = tf.TensorSpec(support_labels_one_hot_shape, tf.float32, name="support_labels_one_hot")
    support_labels_name_spec = tf.TensorSpec(support_labels_name_shape, tf.string, name="support_labels_name")

    def __init__(self, kernel, **kwargs):
        """
        Args:
            support_tensors (tf.Tensor): support set embeddings with shape (n, *embedding_shape)
            support_labels (tf.Tensor): one-hot encoded support set labels with shape (n, n classes)
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.support_tensors = tf.Variable(
            [[]], validate_shape=False, shape=self.support_tensors_shape, name="support_tensors", trainable=False
        )
        self.support_labels_name = tf.Variable(
            [],
            validate_shape=False,
            shape=self.support_labels_name_shape,
            name="support_labels_name",
            dtype=self.support_labels_name_spec.dtype,
            trainable=False,
        )
        self.support_labels_one_hot = tf.Variable(
            [[]], validate_shape=False, shape=self.support_labels_one_hot_shape, name="support_labels_one_hot", trainable=False
        )
        self.columns = tf.Variable([], validate_shape=False, shape=[None], dtype=tf.string, name="columns", trainable=False)
        self.support_set_loss = tf.Variable(0.0, name="support_set_loss", trainable=False)

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

    @tf.function(input_signature=())
    def get_support_set(self):
        return self.support_tensors, self.support_labels_one_hot, self.support_set_loss

    @tf.function(input_signature=(support_tensors_spec, support_labels_name_spec, tf.TensorSpec(None, tf.bool, name="overwrite")))
    def set_support_set(self, support_tensors, support_labels_name, overwrite):
        self._validate_support_set_shape(support_tensors, support_labels_name)
        support_tensors = tf.cond(
            overwrite, lambda: support_tensors, lambda: tf.concat([self.support_tensors, support_tensors], axis=0)
        )
        support_labels_name = tf.cond(
            overwrite, lambda: support_labels_name, lambda: tf.concat([self.support_labels_name, support_labels_name], axis=0),
        )
        columns, codes = tf.unique(support_labels_name)
        support_labels_one_hot = tf.one_hot(codes, depth=tf.size(columns))
        normalized_labels_one_hot = tf.math.divide_no_nan(support_labels_one_hot, tf.reduce_sum(support_labels_one_hot, axis=0))
        self.support_tensors.assign(support_tensors)
        self.support_labels_name.assign(support_labels_name)
        self.support_labels_one_hot.assign(normalized_labels_one_hot)
        self.columns.assign(columns)
        return tf.expand_dims(self.support_set_loss, axis=0)
