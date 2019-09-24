#%%
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import tensorflow.keras.backend as K
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.python.keras.optimizer_v2.adam import Adam

from keras_fsl.models import SiameseNets
from keras_fsl.sequences import RandomBalancedPairsSequence, ProtoNetsSequence, DeterministicSequence, ProductSequence

#%% Init data
all_annotations = (
    pd.read_csv('data/annotations/cropped_images.csv')
    .assign(
        day=lambda df: df.image_name.str.slice(3, 11),
        image_name=lambda df: 'data/cropped_images/' + df.image_name,
    )
)
train_val_test_split = yaml.safe_load(open('data/annotations/train_val_test_split.yaml', 'r'))
train_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['train_set_dates'])]
val_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['val_set_dates'])]
test_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['test_set_dates'])].reset_index(drop=True)
preprocessing = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-180, 180)),
    iaa.CropToFixedSize(224, 224, position='center'),
    iaa.PadToFixedSize(224, 224, position='center'),
    iaa.AssertShape((None, 224, 224, 3)),
    iaa.Lambda(lambda images_list, *_: preprocess_input(np.stack(images_list), data_format='channels_last')),
])

# %% Train Siamese net mixed norms
train_sequence = RandomBalancedPairsSequence(train_set, preprocessing=preprocessing, batch_size=16)
val_sequence = RandomBalancedPairsSequence(val_set, preprocessing=preprocessing, batch_size=16)
output_path = Path('logs') / 'totem_content_crops' / 'siamese_mixed_norms'
output_path.mkdir(parents=True, exist_ok=True)
siamese_nets = SiameseNets(
    branch_model={
        'name': 'ResNet50',
        'init': {'include_top': False, 'input_shape': (224, 224, 3)}
    },
    head_model={
        'name': 'MixedNorms',
        'init': {
            'norms': [
                lambda x: 1 - tf.nn.l2_normalize(x[0]) * tf.nn.l2_normalize(x[1]),
                lambda x: tf.math.abs(x[0] - x[1]),
                lambda x: tf.nn.softmax(tf.math.abs(x[0] - x[1])),
                lambda x: tf.square(x[0] - x[1]),
            ]
        }
    }
)
callbacks = [
    TensorBoard(output_path),
    ModelCheckpoint(
        str(output_path / 'best_model.h5'),
        save_best_only=True,
    ),
    ReduceLROnPlateau(),
    LearningRateScheduler(lambda epoch, lr: 1e-5 if epoch > 10 else 1e-4),
]

optimizer = Adam(lr=1e-4)
siamese_nets.get_layer('branch_model').trainable = False
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=10,
    use_multiprocessing=True,
    workers=10,
)

siamese_nets.get_layer('branch_model').trainable = True
branch_depth = len(siamese_nets.get_layer('branch_model').layers)
for layer in siamese_nets.get_layer('branch_model').layers[:int(branch_depth * 0.8)]:
    layer.trainable = False

K.set_value(optimizer.lr, 1e-5)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=10,
    epochs=30,
    use_multiprocessing=True,
    workers=10,
)

for layer in siamese_nets.get_layer('branch_model').layers[int(branch_depth * 0.5):]:
    layer.trainable = False

K.set_value(optimizer.lr, 1e-5)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=30,
    epochs=50,
    use_multiprocessing=True,
    workers=10,
)

# %% Eval on test set
k_shot = 3
n_way = 5
n_episode = 100
test_sequence = DeterministicSequence(test_set, preprocessing=preprocessing, batch_size=16)
embeddings = siamese_nets.get_layer('branch_model').predict_generator(test_sequence)

scores = []
for _ in range(n_episode):
    selected_labels = np.random.choice(test_set.label.unique(), size=n_way, replace=True)
    support_set = (
        test_set
        .loc[lambda df: df.label.isin(selected_labels)]
        .groupby('label')
        .apply(lambda group: group.sample(k_shot))
        .reset_index('label', drop=True)
    )
    query_set = (
        test_set
        .loc[lambda df: df.label.isin(selected_labels)]
        .loc[lambda df: ~df.index.isin(support_set.index)]
    )
    support_set_embeddings = embeddings[support_set.index]
    query_set_embeddings = embeddings[query_set.index]
    test_sequence = ProductSequence(
        support_images_array=support_set_embeddings,
        query_images_array=query_set_embeddings,
        support_labels=support_set.label.values,
        query_labels=query_set.label.values,
    )
    scores += [(
        test_sequence.pairs_indexes
        .assign(score=siamese_nets.get_layer('head_model').predict_generator(test_sequence, verbose=1))
        .groupby('query_index')
        .apply(lambda group: (
            group
            .sort_values('score', ascending=False)
            .assign(
                average_precision=lambda df: df.target.expanding().mean(),
                good_prediction=lambda df: df.target.iloc[0],
            )
            .loc[lambda df: df.target]
            .agg('mean')
        ))
        .agg('mean')
    )]

scores = pd.DataFrame(scores)[['score', 'average_precision', 'good_prediction']]
scores.to_csv(output_path / 'scores.csv', index=False)

# %% Train Siamese net ProtoNets
train_sequence = ProtoNetsSequence(train_set, preprocessing=preprocessing, batch_size=16)
val_sequence = ProtoNetsSequence(val_set, preprocessing=preprocessing, batch_size=16)
output_path = Path('logs') / 'totem_content_crops' / 'siamese_proto_nets'
output_path.mkdir(parents=True, exist_ok=True)
siamese_nets = SiameseNets(
    branch_model={
        'name': 'ResNet50',
        'init': {'include_top': False, 'input_shape': (224, 224, 3)}
    },
    head_model={
        'name': 'ProtoNets',
        'init': {'k_shot': 3, 'n_way': 5}
    },
)
callbacks = [
    TensorBoard(output_path),
    ModelCheckpoint(
        str(output_path / 'best_model.h5'),
        save_best_only=True,
    ),
    ReduceLROnPlateau(),
]

branch_depth = len(siamese_nets.get_layer('branch_model').layers)
for layer in siamese_nets.get_layer('branch_model').layers[:int(branch_depth * 0.9)]:
    layer.trainable = False

optimizer = Adam(lr=1e-4)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=10,
    workers=0,
)

for layer in siamese_nets.get_layer('branch_model').layers[int(branch_depth * 0.8):]:
    layer.trainable = True

siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=10,
    epochs=20,
    workers=0,
)

for layer in siamese_nets.get_layer('branch_model').layers[int(branch_depth * 0.5):]:
    layer.trainable = True

siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=20,
    epochs=30,
    workers=0,
)

optimizer = Adam(lr=1e-5)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=30,
    epochs=50,
    workers=0,
)
