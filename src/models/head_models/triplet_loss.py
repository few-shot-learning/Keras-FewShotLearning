from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers import Input, Lambda


def TripletLoss(input_shape, margin=0.1, *args, **kwargs):
    """
    Compute the triplet loss between the query tensor and the support set
    """
    query = Input(input_shape)
    axis = list(range(1, len(query.shape)))
    support = [Input(input_shape), Input(input_shape)]
    loss = Lambda(lambda inputs: K.maximum(
        (
            K.sum(K.square(inputs[0] - inputs[1]), axis=axis) -
            K.sum(K.square(inputs[0] - inputs[2]), axis=axis) +
            margin
        ), 0))([query, *support])
    return Model([query, *support], loss, *args, **kwargs)
