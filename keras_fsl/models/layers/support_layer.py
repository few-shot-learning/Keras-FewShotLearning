import tensorflow as tf
from tensorflow.keras.layers import Layer

from keras_fsl.models import head_models


class SupportLayer(Layer):
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def build(self, input_shape):
        if isinstance(input_shape, list):
            embedding_shape = input_shape[0]
        else:
            embedding_shape = input_shape

        if not isinstance(self.kernel, Layer):
            kernel_config = self.kernel
            if isinstance(kernel_config, str):
                kernel_config = {"name": kernel_config}
            kernel_config["init"] = {
                **kernel_config.get("init", {}),
                "input_shape": embedding_shape[1:],
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

    def build_support_set(self, inputs):
        raise NotImplementedError

    @tf.function
    def call(self, inputs, **kwargs):
        support_tensors = self.build_support_set(inputs)
        support_set_size = tf.shape(support_tensors)[0]
        if isinstance(inputs, list):
            embeddings = inputs[0]
        else:
            embeddings = inputs
        return tf.reshape(
            self.kernel(
                [
                    tf.reshape(tf.tile(embeddings, [1, support_set_size]), [-1, tf.shape(embeddings)[1]], name="tf.repeat"),
                    tf.tile(support_tensors, [tf.shape(embeddings)[0], 1]),
                ]
            ),
            [tf.shape(embeddings)[0], support_set_size],
        )