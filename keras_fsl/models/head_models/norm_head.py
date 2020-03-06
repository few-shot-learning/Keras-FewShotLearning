from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda

from ..head_models import norms


def Norm(input_shape, norm="l2", *args, **kwargs):
    query = Input(input_shape)
    support = Input(input_shape)
    loss = Lambda(norms.__dict__[norm])([query, support])
    return Model([query, support], loss, *args, **kwargs)
