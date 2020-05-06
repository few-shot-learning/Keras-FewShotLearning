import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

from keras_fsl.models import encoders, head_models
from keras_fsl.models import layers


def SiameseDetector(
    encoder="Darknet7",
    head_model="DenseSigmoid",
    *args,
    support_input_shape=None,
    pooling_layer="CenterSlicing2D",
    weights=None,
    **kwargs,
):
    """
    Builder for Siamese detector models, see also
    [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/pdf/1606.09549.pdf).

    The idea is to train a Siamese network to match the support image not against the whole support image but only
    against a part of it.

    Args:
        encoder (Union[str, dict, tf.keras.Model]): the branch model of the Siamese net
        head_model (Union[str, dict, tf.keras.Model]): the head model, ie the learnt metric, of the Siamese net
        *args (list): all others args are passed to tf.keras.Model
        support_input_shape (tuple): input_shape of the support image. By default the input shape of the query image
            is the one given by the encoder.input_shape and the support_input_shape is taken to be the same as the
            query_input_shape (usual Siamese Net classifier).
        pooling_layer (Union[str, dict, tf.keras.Model]): the layer to be used atop the support embedding so as to
            reduce it to a one-dimensional embedding; e.g. GlobalAveragePooling2D or CenterSlicing2D
        weights (str): path to weights to be loaded at the end
        **kwargs (dict): all other kwargs are passed to tf.keras.Model

    Returns:

    """
    if not isinstance(encoder, Model):
        if isinstance(encoder, str):
            encoder = {"name": encoder}
        encoder_name = encoder["name"]
        encoder = getattr(encoders, encoder_name)(**encoder.get("init", {}))
    encoder = Model(encoder.inputs, encoder.outputs, name="encoder")
    support_input_shape = support_input_shape or encoder.input_shape[1:]

    embedding_dimension = encoder.output_shape[-1]
    output_query_grid_size = np.array(encoder.output_shape[1:3])

    if not isinstance(head_model, Model):
        if isinstance(head_model, str):
            head_model = {"name": head_model}
        head_model_name = head_model["name"]
        head_model_init = {
            **head_model.get("init", {}),
            "input_shape": (embedding_dimension,),
        }
        head_model = getattr(head_models, head_model_name)(**head_model_init)
    head_model = Model(head_model.inputs, head_model.outputs, name="head_model")

    if not isinstance(pooling_layer, Model):
        if isinstance(pooling_layer, str):
            pooling_layer = {"name": pooling_layer}
        pooling_layer_name = pooling_layer["name"]
        pooling_layer = getattr(layers, pooling_layer_name)(**pooling_layer.get("init", {}))

    query = Input(shape=encoder.input_shape[1:], name="query")
    supports = [Input(shape=support_input_shape, name=f"support_{i}") for i in range(1, len(head_model.inputs))]

    query_embedding = Lambda(lambda x: tf.reshape(x, (-1, embedding_dimension)))(encoder(query))
    supports_embeddings = [
        Lambda(lambda x: tf.tile(x, (output_query_grid_size.prod(), 1)))(pooling_layer(encoder(input_))) for input_ in supports
    ]

    scores = head_model([query_embedding, *supports_embeddings])

    output = Lambda(lambda x: tf.reshape(x, (-1, *output_query_grid_size)))(scores)

    model = Model([query, *supports], output, *args, **kwargs)
    if weights is not None:
        model.load_weights(weights)

    return model
