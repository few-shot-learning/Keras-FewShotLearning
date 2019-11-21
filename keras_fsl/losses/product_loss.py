import tensorflow as tf

from tensorflow.python.keras.layers import Layer


class ProductLoss(Layer):
    """
    A layer that takes as input a batch of embeddings, their corresponding labels and a multi-inputs loss and compute
    all the possible losses of each batch according to this
    multi-inputs loss. For instance if the input loss is the L2 norm between two tensors, then this loss will evaluate
    the L2 distance between all pairs of tensors of a given batch.
    """

    def __init__(self, loss, metric_layer, target_function, **kwargs):
        """

        Args:
            loss (Union[str, tensorflow.keras.losses.Loss): the actual loss computed on the output of the loss layer
            metric_layer (tensorflow.keras.layers.Layer): The multi-inputs model computing the loss
            target_function (function): a function to be applied to the corresponding labels to compute the target of
                the loss
        """
        super().__init__(**kwargs)
        self.loss = loss
        self.metric_layer = metric_layer
        self.target_function = target_function
        self.indexes = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': self.loss,
            'metric_layer': self.metric_layer,
            'target_function': self.target_function,
        })
        return config

    def build(self, input_shape):
        batch_size = input_shape[0][0] or input_shape[1][0]
        self.indexes = tf.range(batch_size, dtype=tf.int32)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[1]**2,

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (List[tf.Tensor]): embeddings and labels of a batch. Embeddings with shape
                (batch_size, *embedding_shape) and labels as multi-class categorical tensor, e.g [1, 0, 2, 3]
        """
        embeddings, labels = inputs
        indexes = tf.meshgrid(*[
            self.indexes for _ in range(len(self.metric_layer.inputs))
        ])
        loss = self.loss(
            tf.reshape(self.target_function([
                tf.gather(labels, tf.reshape(indexes[i], [-1])) for i in range(len(indexes))
            ]), [-1]),
            tf.reshape(self.metric_layer([
                tf.gather(embeddings, tf.reshape(indexes[i], [-1])) for i in range(len(indexes))
            ]), [-1]),
        )
        self.add_loss(tf.reduce_mean(loss), inputs=True)
        return loss
