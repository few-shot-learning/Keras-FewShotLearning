
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model


def DenseSigmoid(input_shape):
    """
    Add a Dense layer on top of the coordinate-wise abs difference between the embeddings
    """
    query = Input(input_shape)
    support = Input(input_shape)
    abs_difference = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([query, support])
    prediction = Dense(1, activation='sigmoid')(abs_difference)
    return Model(inputs=[query, support], outputs=prediction)
