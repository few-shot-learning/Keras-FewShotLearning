from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input

from src.models import branch_models, head_models


def SiameseNets(
    branch_model='ResNet50',
    head_model='Norm',
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

    if not isinstance(head_model, Model):
        if isinstance(head_model, str):
            head_model = {'name': head_model}
        head_model_name = head_model['name']
        head_model_init = {
            **head_model.get('init', {}),
            'input_shape': branch_model.output.shape[1:],
        }
        head_model = getattr(head_models, head_model_name)(**head_model_init)
    head_model = Model(head_model.inputs, head_model.outputs, name='head_model')

    inputs = [Input(shape=branch_model.input_shape[1:], name=f'input_{i}') for i in range(len(head_model.inputs))]
    embeddings = [branch_model(input_) for input_ in inputs]
    output = head_model(embeddings)

    model = Model(inputs, output, *args, **kwargs)
    if weights is not None:
        model.load_weights(weights)

    return model
