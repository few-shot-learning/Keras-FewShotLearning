import tensorflow as tf
from tensorflow.keras.layers import Activation, Layer


class CentroidsSimilarity(Layer):
    """
    Compute the similarity (distance) between all items of the input batch and the centroid of each class found in the batch.
    """

    def __init__(self, kernel, activation="linear", **kwargs):
        """

        Args:
            kernel: similarity / distance function (x, x') => tf.float32
            activation: add an activation on top of the distances to centroid, (e.g. softmax to compute ProtoNets output). The arg is
                directly passed to tf.keras.layers.Activation layer.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.activation = Activation(activation)

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
    def call(self, inputs):
        if not isinstance(inputs, list) or isinstance(inputs, list) and len(inputs) != 2:
            raise AttributeError("Layer should be called on a list of tensors [embeddings, y_true]")
        embeddings, y_true = inputs
        centroids = tf.matmul(tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=0)), embeddings, transpose_a=True)
        return self.activation(
            tf.reshape(
                self.kernel(
                    [
                        tf.reshape(
                            tf.tile(embeddings, [1, tf.shape(centroids)[0]]), [-1, tf.shape(embeddings)[1]], name="tf.repeat"
                        ),
                        tf.tile(centroids, [tf.shape(embeddings)[0], 1]),
                    ]
                ),
                [tf.shape(embeddings)[0], tf.shape(centroids)[0]],
            )
        )
