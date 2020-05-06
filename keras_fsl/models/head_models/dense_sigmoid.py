import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model


def DenseSigmoid(input_shape, use_bias=True):
    """
    Add a Dense layer on top of the coordinate-wise abs difference between the embeddings.
    Similar to original [SiameseNets paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
    """
    query = Input(input_shape)
    support = Input(input_shape)
    output = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([query, support])
    output = Dense(1, activation="sigmoid", use_bias=use_bias)(output)
    return Model(inputs=[query, support], outputs=output)
