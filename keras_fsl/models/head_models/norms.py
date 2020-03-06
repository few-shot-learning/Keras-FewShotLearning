import tensorflow as tf


@tf.function
def l1(inputs):
    """
    Compute the l1 norm between two tensors, batch wise
    """
    return tf.expand_dims(tf.reduce_sum(tf.abs(inputs[0] - inputs[1]), axis=list(range(1, len(inputs[0].shape)))), 1)


@tf.function
def l2(inputs):
    """
    Compute the l2 norm between two tensors, batch wise
    """
    return tf.expand_dims(tf.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=list(range(1, len(inputs[0].shape)))), 1)


@tf.function
def linf(inputs):
    """
    Compute the linf norm between two tensors, batch wise
    """
    return tf.expand_dims(tf.reduce_max(tf.abs(inputs[0] - inputs[1]), axis=list(range(1, len(inputs[0].shape)))), 1)


@tf.function
def cosine_similarity(inputs):
    """
    Compute the cosine similarity between two tensors, batch wise
    """
    axis = list(range(1, len(inputs[0].shape)))
    return tf.expand_dims(
        1
        - tf.reduce_sum(
            tf.nn.l2_normalize(inputs[0], axis=axis) * tf.nn.l2_normalize(inputs[1], axis=axis), axis=axis,
        ),
        1,
    )
