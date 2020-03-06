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
    abs_difference = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([query, support])
    prediction = Dense(1, activation="sigmoid", use_bias=use_bias)(abs_difference)
    return Model(inputs=[query, support], outputs=prediction)
