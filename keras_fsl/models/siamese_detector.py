import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

from keras_fsl.models import branch_models, head_models
from keras_fsl.models import layers


def SiameseDetector(
    branch_model='Darknet7',
    head_model='DenseSigmoid',
    *args,
    support_input_shape=None,
    pooling_layer='CenterSlicing2D',
    weights=None,
    **kwargs,
):
    """
    Builder for Siamese detector models, see also
    [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/pdf/1606.09549.pdf).

    The idea is to train a Siamese network to match the support image not against the whole support image but only
    against a part of it.

    Args:
        branch_model (Union[str, dict, tf.keras.Model]): the branch model of the Siamese net
        head_model (Union[str, dict, tf.keras.Model]): the head model, ie the learnt metric, of the Siamese net
        *args (list): all others args are passed to tf.keras.Model
        support_input_shape (tuple): input_shape of the support image. By default the input shape of the query image
            is the one given by the branch_model.input_shape and the support_input_shape is taken to be the same as the
            query_input_shape (usual Siamese Net classifier).
        pooling_layer (Union[str, dict, tf.keras.Model]): the layer to be used atop the support embedding so as to
            reduce it to a one-dimensional embedding; e.g. GlobalAveragePooling2D or CenterSlicing2D
        weights (str): path to weights to be loaded at the end
        **kwargs (dict): all other kwargs are passed to tf.keras.Model

    Returns:

    """
    if not isinstance(branch_model, Model):
        if isinstance(branch_model, str):
            branch_model = {'name': branch_model}
        branch_model_name = branch_model['name']
        branch_model = getattr(branch_models, branch_model_name)(**branch_model.get('init', {}))
    branch_model = Model(branch_model.inputs, branch_model.outputs, name='branch_model')
    support_input_shape = support_input_shape or branch_model.input_shape[1:]

    embedding_dimension = branch_model.output_shape[-1]
    output_query_grid_size = np.array(branch_model.output_shape[1:3])

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

    if not isinstance(pooling_layer, Model):
        if isinstance(pooling_layer, str):
            pooling_layer = {'name': pooling_layer}
        pooling_layer_name = pooling_layer['name']
        pooling_layer = getattr(layers, pooling_layer_name)(**pooling_layer.get('init', {}))

    query = Input(shape=branch_model.input_shape[1:], name='query')
    supports = [Input(shape=support_input_shape, name=f'support_{i}') for i in range(1, len(head_model.inputs))]

    query_embedding = Lambda(lambda x: tf.reshape(x, (-1, embedding_dimension)))(branch_model(query))
    supports_embeddings = [
        Lambda(lambda x: tf.tile(x, (output_query_grid_size.prod(), 1)))(pooling_layer(branch_model(input_)))
        for input_ in supports
    ]

    scores = head_model([query_embedding, *supports_embeddings])

    output = Lambda(lambda x: tf.reshape(x, (-1, *output_query_grid_size)))(scores)

    model = Model([query, *supports], output, *args, **kwargs)
    if weights is not None:
        model.load_weights(weights)

    return model
