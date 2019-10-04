import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Lambda, Flatten

from keras_fsl.models import branch_models, head_models


def SiameseDetector(
    branch_model='MobileNet',
    head_model='DenseSigmoid',
    *args,
    weights=None,
    **kwargs,
):
    if not isinstance(branch_model, Model):
        if isinstance(branch_model, str):
            branch_model = {'name': branch_model}
        branch_model_name = branch_model['name']
        branch_model = getattr(branch_models, branch_model_name)(**branch_model.get('init', {}))
    branch_model = Model(branch_model.inputs, branch_model.outputs, name='branch_model')
    embedding_dimension = branch_model.output_shape[-1]
    input_grid_size = np.array(branch_model.input_shape[1:3])
    output_grid_size = np.array(branch_model.output_shape[1:3])
    catalog_input_shape = (*(input_grid_size // output_grid_size), 3)

    if not isinstance(head_model, Model):
        if isinstance(head_model, str):
            head_model = {'name': head_model}
        head_model_name = head_model['name']
        head_model_init = {
            **head_model.get('init', {}),
            'input_shape': (embedding_dimension,),
        }
        head_model = getattr(head_models, head_model_name)(**head_model_init)
    head_model = Model(head_model.inputs, head_model.outputs, name='head_model')

    query = Input(shape=branch_model.input_shape[1:], name='query')
    supports = [Input(shape=catalog_input_shape, name=f'support_{i}') for i in range(1, len(head_model.inputs))]
    query_embedding = Lambda(lambda x: tf.reshape(x, (-1, embedding_dimension)))(branch_model(query))
    supports_embeddings = [
        Lambda(lambda x: tf.tile(x, (output_grid_size.prod(), 1)))(Flatten()(branch_model(input_)))
        for input_ in supports
    ]
    output = head_model([query_embedding, *supports_embeddings])

    model = Model([query, *supports], output, *args, **kwargs)
    if weights is not None:
        model.load_weights(weights)

    return model
