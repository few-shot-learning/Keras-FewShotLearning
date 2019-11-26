import tensorflow as tf

from tensorflow.python.keras.layers import Layer


class ClassificationLoss(Layer):

    def __init__(self, loss, similarity_layer, **kwargs):
        """
        Args:
            loss: categorical loss function to be used on the predictions
            similarity_layer: function to compute pair-wise similarity. Score is assumed to be normalized between 0 and 1 and
                1 meaning same and 0 different. This similarity can be learnt during training when passed a layer.
        """
        super().__init__(**kwargs)
        self.loss = loss
        self.similarity_layer = similarity_layer
        if len(self.similarity_layer.inputs) != 2:
            raise NotImplementedError('This loss currently supports only similarity_layer with two inputs')
        self.indexes = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': self.loss,
            'similarity_layer': self.similarity_layer,
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
            self.indexes for _ in range(len(self.similarity_layer.inputs))
        ])
        scores_matrix = tf.reshape(self.similarity_layer([
            tf.gather(embeddings, tf.reshape(indexes[i], [-1])) for i in range(len(indexes))
        ]), [len(self.indexes), len(self.indexes)])
        scores_matrix = tf.linalg.set_diag(scores_matrix, [0] * len(self.indexes))  # do not select same image score
        classes, encoding = tf.unique(tf.reshape(labels, [-1]))
        labels_one_hot = tf.one_hot(encoding, depth=tf.reduce_max(encoding))
        loss = self.loss(
            labels_one_hot,
            tf.nn.softmax(
                tf.linalg.matmul(scores_matrix, labels_one_hot) / tf.reduce_sum(labels_one_hot, axis=0),
                axis=1,
            ),
        )
        self.add_loss(tf.reduce_mean(loss), inputs=True)
        return loss
