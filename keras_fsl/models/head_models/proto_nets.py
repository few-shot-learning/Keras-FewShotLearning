import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Lambda, Flatten, Activation, Concatenate


def ProtoNets(input_shape, k_shot=5, n_way=5, **kwargs):
    """
    Head model defining a [Protypical networks](https://arxiv.org/pdf/1703.05175.pdf)

    Args:
        input_shape (Union[list, tuple]): input_shape to be passed to Input layer
        k_shot (int): number of images per class in the support set
        n_way: (int): number of classes in the support set
        **kwargs: all other kwargs are passed to tf.norm. Should be used to specify another norm with ord='l1'
            for instance
    """
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
            lambda inputs: tf.norm(inputs[0] - inputs[1], axis=1, keepdims=True, **kwargs), name=f'distance_{n}'
        )([prototype, query_flatten])
        for n, prototype in enumerate(prototypes)
    ])
    output = Activation('softmax')(distances)

    return Model([query, *support], output)
