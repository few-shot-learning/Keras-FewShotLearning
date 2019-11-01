#%%
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from keras_fsl.models import SiameseNets
from keras_fsl.sequences.training.pairs import RandomBalancedPairsSequence, BalancedPairsSequence
from keras_fsl.sequences.training.single import DeterministicSequence
from keras_fsl.sequences.prediction.pairs import ProductSequence

#%% Init data
output_path = Path('logs') / 'content_crops' / 'siamese_mixed_norms'
output_path.mkdir(parents=True, exist_ok=True)

all_annotations = (
    pd.read_csv('data/annotations/cropped_images.csv')
    .assign(
        day=lambda df: df.image_name.str.slice(3, 11),
        image_name=lambda df: 'data/images/cropped_images/' + df.image_name,
    )
)
train_val_test_split = yaml.safe_load(open('data/annotations/cropped_images_split.yaml', 'r'))
train_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['train_set_dates'])]
val_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['val_set_dates'])]
test_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['test_set_dates'])].reset_index(drop=True)

# %% Init model
preprocessing = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-180, 180)),
    iaa.CropToFixedSize(224, 224, position='center'),
    iaa.PadToFixedSize(224, 224, position='center'),
    iaa.AssertShape((None, 224, 224, 3)),
    iaa.Lambda(lambda images_list, *_: preprocess_input(np.stack(images_list), data_format='channels_last')),
])

siamese_nets = SiameseNets(
    branch_model={
        'name': 'MobileNet',
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

# %% Pre-train branch_model as usual classifier on big classes
callbacks = [
    TensorBoard(output_path / 'branch_model'),
    ModelCheckpoint(
        str(output_path / 'branch_model' / 'best_model.h5'),
        save_best_only=True,
    ),
    ReduceLROnPlateau(),
]
branch_model_train_set = (
    train_set
    .groupby('label')
    .filter(lambda group: len(group) > 100)
)
branch_model_val_set = (
    val_set
    .loc[lambda df: df.label.isin(branch_model_train_set.label.unique())]
)
train_sequence = DeterministicSequence(branch_model_train_set, preprocessings=preprocessing, batch_size=16)
val_sequence = DeterministicSequence(branch_model_val_set, preprocessings=preprocessing, batch_size=16)

branch_classifier = Sequential([
    siamese_nets.get_layer('branch_model'),
    GlobalAveragePooling2D(),
    Dense(len(branch_model_train_set.label.unique())),
])
optimizer = Adam(lr=1e-5)
branch_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy')
branch_classifier.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=10,
    use_multiprocessing=True,
    workers=5,
)

# %% Train model
callbacks = [
    TensorBoard(output_path),
    ModelCheckpoint(
        str(output_path / 'best_model.h5'),
        save_best_only=True,
    ),
    ReduceLROnPlateau(),
]
train_sequence = RandomBalancedPairsSequence(train_set, preprocessings=preprocessing, batch_size=16)
val_sequence = RandomBalancedPairsSequence(val_set, preprocessings=preprocessing, batch_size=16)

siamese_nets.get_layer('branch_model').trainable = False
optimizer = Adam(lr=1e-5)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=5,
    use_multiprocessing=True,
    workers=5,
)

siamese_nets.get_layer('branch_model').trainable = True
branch_depth = len(siamese_nets.get_layer('branch_model').layers)
for layer in siamese_nets.get_layer('branch_model').layers[:int(branch_depth * 0.8)]:
    layer.trainable = False

optimizer = Adam(1e-5)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=5,
    epochs=15,
    use_multiprocessing=True,
    workers=5,
)

for layer in siamese_nets.get_layer('branch_model').layers[int(branch_depth * 0.5):]:
    layer.trainable = False

optimizer = Adam(1e-5)
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

train_sequence = BalancedPairsSequence(train_set, pairs_per_query=5, preprocessings=preprocessing, batch_size=16)
val_sequence = BalancedPairsSequence(val_set, pairs_per_query=5, preprocessings=preprocessing, batch_size=16)
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
n_way = 10
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
