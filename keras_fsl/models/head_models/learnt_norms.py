import numpy as np
from tensorflow.keras import activations
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model


def LearntNorms(input_shape, use_bias=True, activation="sigmoid"):
    """
    Learn the coordinate-wise comparison as opposed to the MixedNorms where they are provided as input
    Args:
        input_shape (tuple): arg to be passed to keras.layer.Input
        use_bias (bool), whether to use bias in layers or not
        activation: add an activation function as in other keras standard layers. Default to sigmoid to output a normalize score
    """
    embedding_dimension = np.prod(input_shape)
    query = Input(input_shape)
    support = Input(input_shape)
    inputs = [query, support]
    output = Concatenate(axis=1)(inputs)
    output = Reshape((len(inputs), embedding_dimension, 1), name="stack")(output)

    output = Conv2D(filters=32, kernel_size=(len(inputs), 1), activation="relu", name="norms_creation", use_bias=use_bias)(
        output
    )
    output = Conv2D(filters=1, kernel_size=(1, 1), activation="linear", name="norms_average", use_bias=use_bias)(output)
    output = Flatten()(output)

    output = Dense(1, activation=activations.get(activation), name="output", use_bias=use_bias)(output)
    return Model(inputs=inputs, outputs=output)
