import tensorflow as tf
from tensorflow.keras.layers import Layer


class KernelMatrix(Layer):
    """
    Compute the kernel matrix of the input batch using the given function. Note that this function can be learnt (if
    given a model for instance).
    """

    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.batch_size = None

    def build(self, input_shape):
        batch_size = input_shape[0]
        if batch_size is None:
            raise AttributeError('Layer has to be built with a proper batch_size')
        self.batch_size = batch_size
        super().build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                raise AttributeError('Layer should be called on a single tensor')
            inputs = inputs[0]
        return tf.reshape(
            self.kernel([
                tf.reshape(tf.tile(inputs, [1, self.batch_size]), [-1, inputs.shape[1]], name='tf.repeat'),
                tf.tile(inputs, [self.batch_size, 1]),
            ]),
            [self.batch_size, self.batch_size],
        )

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.batch_size
