import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Lambda, Flatten, Activation, Concatenate


def ProtoNets(input_shape, k_shot=5, n_way=5):
    query = Input(input_shape, name='query')
    support = [
        Input(input_shape, name=f'support_{n}_shot_{k}')
        for n in range(n_way)
        for k in range(k_shot)
    ]
    prototypes = [
        Flatten()(
            Lambda(
                lambda shots: tf.reduce_mean(tf.stack(shots, axis=-1), axis=-1), name=f'prototype_{n}'
            )(support[(n * k_shot):((n+1) * k_shot)])
        )
        for n in range(n_way)
    ]
    query_flatten = Flatten()(query)
    distances = Concatenate(axis=1)([
        Lambda(
            lambda inputs: tf.norm(inputs[0] - inputs[1], axis=1, keepdims=True), name=f'distance_{n}'
        )([prototype, query_flatten])
        for n, prototype in enumerate(prototypes)
    ])
    output = Activation('softmax')(distances)

    return Model([query, *support], output)
