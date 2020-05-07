import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Reshape,
)
from tensorflow.keras.models import Model


def MixedNorms(input_shape, norms=None, use_bias=True, activation="sigmoid"):
    """
    Head inspired by
    [this kaggle notebook](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563)
    on whale identification

    Args:
        input_shape (tuple): arg to be passed to keras.layer.Input
        norms (List[function]): list of function to be applied to the list of tensors [query, support] in a Lambda layer
        use_bias (bool), whether to use bias in layers or not
        activation: add an activation function as in other keras standard layers. Default to sigmoid to output a normalize score
    """
    if norms is None:
        norms = [
            lambda x: x[0] * x[1],
            lambda x: x[0] + x[1],
            lambda x: tf.math.abs(x[0] - x[1]),
            lambda x: tf.math.square(x[0] - x[1]),
        ]

    query = Input(input_shape)
    support = Input(input_shape)
    inputs = [query, support]
    if len(input_shape) == 3:
        inputs = [GlobalAveragePooling2D()(input_) for input_ in inputs]

    output = Concatenate()([Lambda(norm)(inputs) for norm in norms])
    output = Reshape((len(norms), input_shape[-1], 1), name="stack")(output)

    output = Conv2D(filters=32, kernel_size=(len(norms), 1), activation="relu", name="norms_selection", use_bias=use_bias)(
        output
    )
    output = Conv2D(filters=1, kernel_size=(1, 1), activation="linear", name="norms_average", use_bias=use_bias)(output)
    output = Flatten()(output)

    output = Dense(1, activation=activations.get(activation), name="output", use_bias=use_bias)(output)
    return Model(inputs=inputs, outputs=output)
