import tensorflow as tf
from tensorflow.keras.layers import Layer


class GramMatrix(Layer):
    """
    Compute the gram matrix of the input batch using the given function. Note that this function can be learnt (if
    given a layer for instance).
    """

    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def get_config(self):
        config = super().get_config()
        config.update({"kernel": self.kernel.to_json()})
        return config

    @classmethod
    def from_config(cls, config):
        kernel = tf.keras.models.model_from_json(config["kernel"])
        config["kernel"] = kernel
        return cls(**config)

    @tf.function
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                raise AttributeError("Layer should be called on a single tensor")
            inputs = inputs[0]
        return tf.reshape(
            self.kernel(
                [
                    tf.reshape(tf.tile(inputs, [1, tf.shape(inputs)[0]]), [-1, tf.shape(inputs)[1]], name="tf.repeat"),
                    tf.tile(inputs, [tf.shape(inputs)[0], 1]),
                ]
            ),
            [tf.shape(inputs)[0], tf.shape(inputs)[0]],
        )
