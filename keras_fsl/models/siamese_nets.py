from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from keras_fsl.models import encoders, head_models


def SiameseNets(
    encoder="KochNet", head_model="DenseSigmoid", *args, weights=None, **kwargs,
):
    if not isinstance(encoder, Model):
        if isinstance(encoder, str):
            encoder = {"name": encoder}
        encoder_name = encoder["name"]
        encoder = getattr(encoders, encoder_name)(**encoder.get("init", {}))
    encoder = Model(encoder.inputs, encoder.outputs, name="encoder")

    if not isinstance(head_model, Model):
        if isinstance(head_model, str):
            head_model = {"name": head_model}
        head_model_name = head_model["name"]
        head_model_init = {
            **head_model.get("init", {}),
            "input_shape": encoder.output.shape[1:],
        }
        head_model = getattr(head_models, head_model_name)(**head_model_init)
    head_model = Model(head_model.inputs, head_model.outputs, name="head_model")

    inputs = [Input(shape=encoder.input_shape[1:], name=f"input_{i}") for i in range(len(head_model.inputs))]
    embeddings = [encoder(input_) for input_ in inputs]
    output = head_model(embeddings)

    model = Model(inputs, output, *args, **kwargs)
    if weights is not None:
        model.load_weights(weights)

    return model
