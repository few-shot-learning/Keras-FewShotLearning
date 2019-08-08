from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers import Input, Lambda


def L1(input_shape, *args, **kwargs):
    """
    Compute the l1 norm between two tensors
    """
    query = Input(input_shape)
    support = Input(input_shape)
    loss = Lambda(lambda inputs: (
        K.sum(K.abs(inputs[0] - inputs[1]), axis=list(range(1, len(query.shape))))
    ))([query, support])
    return Model([query, support], loss, *args, **kwargs)


def L2(input_shape, *args, **kwargs):
    """
    Compute the squared l2 norm between two tensors
    """
    query = Input(input_shape)
    support = Input(input_shape)
    loss = Lambda(lambda inputs: (
        K.sum(K.square(inputs[0] - inputs[1]), axis=list(range(1, len(query.shape))))
    ))([query, support])
    return Model([query, support], loss, *args, **kwargs)


def LInf(input_shape, *args, **kwargs):
    """
    Compute the infinite norm between two tensors
    """
    query = Input(input_shape)
    support = Input(input_shape)
    loss = Lambda(lambda inputs: (
        K.max(K.abs(inputs[0] - inputs[1]), axis=list(range(1, len(query.shape))))
    ))([query, support])
    return Model([query, support], loss, *args, **kwargs)


def CosineSimilarity(input_shape, *args, **kwargs):
    """
    Compute the squared l2 norm between two tensors
    """
    query = Input(input_shape)
    support = Input(input_shape)
    axis = list(range(1, len(query.shape)))
    loss = Lambda(lambda inputs: (
        1 - K.sum(K.l2_normalize(inputs[0], axis=axis) * K.l2_normalize(inputs[1], axis=axis), axis=axis)
    ))([query, support])
    return Model([query, support], loss, *args, **kwargs)
