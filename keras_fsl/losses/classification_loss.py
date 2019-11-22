import tensorflow as tf

from tensorflow.python.keras.layers import Layer


class ClassificationLoss(Layer):

    def __init__(self, loss, metric_layer, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.metric_layer = metric_layer
        if len(self.metric_layer.inputs) != 2:
            raise NotImplementedError('This loss currently supports only metric_later with two inputs')
        self.indexes = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': self.loss,
            'metric_layer': self.metric_layer,
        })
        return config

    def build(self, input_shape):
        batch_size = input_shape[0][0] or input_shape[1][0]
        self.indexes = tf.range(batch_size, dtype=tf.int32)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return len(self.indexes),

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
        scores_matrix = tf.reshape(self.metric_layer([
            tf.gather(embeddings, tf.reshape(indexes[i], [-1])) for i in range(len(indexes))
        ]), [len(self.indexes), len(self.indexes)])
        tf.linalg.set_diag(scores_matrix, [0] * len(self.indexes))  # do not select same image score
        best_match = tf.argmax(scores_matrix, axis=1)
        predicted_labels = tf.gather(labels, best_match)
        loss = self.loss(labels, predicted_labels)
        self.add_loss(tf.reduce_mean(loss), inputs=True)
        return loss
