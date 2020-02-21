from functools import wraps

import pandas as pd
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, ReLU, Concatenate

from keras_fsl.models import branch_models


ANCHORS = pd.DataFrame([
    [0, 116, 90],
    [0, 156, 198],
    [0, 373, 326],
    [1, 30, 61],
    [1, 62, 45],
    [1, 59, 119],
    [2, 10, 13],
    [2, 16, 30],
    [2, 33, 23],
], columns=['scale', 'width', 'height'])


@wraps(Conv2D)
def conv_block(*args, **kwargs):
    return Sequential([
        Conv2D(*args, **kwargs, use_bias=False),
        BatchNormalization(),
        ReLU(),
    ])


def bottleneck(filters, *args, **kwargs):
    return Sequential([
        conv_block(filters // 4, (1, 1), padding='same'),
        conv_block(filters, (3, 3), padding='same'),
    ], *args, **kwargs)


def up_sampling_block(filters, *args, **kwargs):
    return Sequential([
        conv_block(filters, (1, 1), padding='same'),
        UpSampling2D(2),
    ], *args, **kwargs)


def FeaturePyramidNet(
    backbone='MobileNet',
    *args,
    feature_maps=3,
    objectness=True,
    anchors=None,
    classes=None,
    weights=None,
    **kwargs,
):
    """
    Multi scale feature extractor following the [Feature Pyramid Network for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
    framework.

    It analyses the given backbone architecture so as to extract the features maps at relevant positions (last position before downsampling)
    Then it builds a model with as many feature maps (outputs) as requested, starting from the deepest.

    Args:
        backbone (Union[str, dict, tensorflow.keras.Model]): parameters of the feature extractor
        feature_maps (int): number of feature maps to extract from the backbone.
        objectness (bool): whether to add a score for object presence probability or not (similar to add a background class, see Yolo for
            instance).
        anchors (pandas.DataFrame): containing scale, width and height columns. Scale column will be used to select the corresponding
            feature map: 0 for the smallest resolution, 1 for the next one, etc.
        classes (pandas.Series): provide classes to build a single-shot detector from the anchors and the feature maps.
        weights (Union[str, pathlib.Path]): path to the weights file to load with tensorflow.keras.load_weights
    """
    if not isinstance(backbone, Model):
        if isinstance(backbone, str):
            backbone = {'name': backbone, 'init': {'include_top': False, 'input_shape': (416, 416, 3)}}
        backbone_name = backbone['name']
        backbone = getattr(branch_models, backbone_name)(**backbone.get('init', {}))

    output_shapes = (
        pd.DataFrame([
            layer.input_shape[0]
            if isinstance(layer.input_shape, list)
            else layer.output_shape
            for layer in backbone.layers
        ], columns=['batch_size', 'width', 'height', 'channels'])
        .loc[lambda df: df.width.iloc[0] % df.width == 0]
        .drop_duplicates(['width', 'height'], keep='last')
        .sort_index(ascending=False)
    )

    outputs = []
    for output_shape in output_shapes.iloc[:feature_maps].itertuples():
        input_ = backbone.layers[output_shape.Index].output
        if outputs:
            pyramid_input = up_sampling_block(output_shape.channels, name=f'up_sampling_{output_shape.channels}')(outputs[-1])
            input_ = Concatenate()([input_, pyramid_input])
        outputs += [bottleneck(output_shape.channels, name=f'bottleneck_{output_shape.channels}')(input_)]

    if classes is not None:
        if anchors is None:
            anchors = ANCHORS.copy()
        anchors = anchors.assign(id=lambda df: 'scale_' + df.scale.astype(str) + '_' + df.width.astype(str) + 'x' + df.height.astype(str))
        outputs = [
            Concatenate(axis=3)(
                [Conv2D(4, (1, 1), name=f'{anchor.id}_box')(outputs[anchor.scale])] +
                ([Conv2D(1, (1, 1), name=f'{anchor.id}_objectness')(outputs[anchor.scale])] if objectness else []) +
                [Conv2D(1, (1, 1), name=f'{anchor.id}_{label}')(outputs[anchor.scale]) for label in classes],
            )
            for anchor in anchors.itertuples()
        ]

    model = Model(backbone.input, outputs, *args, **kwargs)
    if weights is not None:
        model.load_weights(weights)

    return model
