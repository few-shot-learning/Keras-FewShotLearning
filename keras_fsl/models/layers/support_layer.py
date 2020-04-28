import tensorflow as tf
from tensorflow.keras.layers import Layer

from keras_fsl.models import head_models


class SupportLayer(Layer):
    def __init__(self, kernel, **kwargs):
        kwargs["dynamic"] = True
        super().__init__(**kwargs)
        self.support_tensors = tf.constant([])
        self.support_labels = tf.constant([])
        self.kernel = kernel

    def build(self, input_shape):
        if not isinstance(self.kernel, Layer):
            kernel_config = self.kernel
            if isinstance(kernel_config, str):
                kernel_config = {"name": kernel_config}
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            kernel_config["init"] = {
                **kernel_config.get("init", {}),
                "input_shape": input_shape[1:],
            }
            self.kernel = getattr(head_models, kernel_config["name"])(**kernel_config["init"])

    def get_config(self):
        config = super().get_config()
        config.update({"kernel": self.kernel.to_json()})
        return config

    @classmethod
    def from_config(cls, config):
        kernel = tf.keras.models.model_from_json(config["kernel"])
        config["kernel"] = kernel
        return cls(**config)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return tf.TensorShape([input_shape[0][0], None])
        return tf.TensorShape([input_shape[0], None])

    def set_support_set(self, inputs):
        if isinstance(inputs, list):
            self.support_tensors = inputs[0]
            if len(inputs) == 2:
                self.support_labels = inputs[1]
        else:
            self.support_tensors = inputs

    def get_support_set(self):
        return self.support_tensors, self.support_labels

    @property
    def _state_size(self):
        return tf.shape(self.support_tensors)[0]

    def call(self, inputs, training=None):
        if training or isinstance(inputs, list) and len(inputs) == 2:
            self.set_support_set(inputs)
        if isinstance(inputs, list):
            embeddings = inputs[0]
        else:
            embeddings = inputs
        return tf.reshape(
            self.kernel(
                [
                    tf.reshape(tf.tile(embeddings, [1, self._state_size]), [-1, tf.shape(embeddings)[1]], name="tf.repeat"),
                    tf.tile(self.support_tensors, [tf.shape(embeddings)[0], 1]),
                ]
            ),
            [tf.shape(embeddings)[0], self._state_size],
        )
