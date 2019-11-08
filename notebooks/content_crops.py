#%%
from datetime import datetime
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import applications as keras_applications
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.optimizer_v2.adam import Adam

from keras_fsl.models import SiameseNets
from keras_fsl.sequences.prediction.pairs import ProductSequence
from keras_fsl.sequences.training.pairs import BalancedPairsSequence, RandomBalancedPairsSequence
from keras_fsl.sequences.training.single import DeterministicSequence

#%% Init data
output_path = Path('logs') / 'content_crops' / 'siamese_mixed_norms' / datetime.today().strftime('%Y%m%d-%H%M%S')
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

#%% Init model
branch_model_name = 'MobileNet'

preprocessing = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-180, 180)),
    iaa.CropToFixedSize(224, 224, position='center'),
    iaa.PadToFixedSize(224, 224, position='center'),
    iaa.AssertShape((None, 224, 224, 3)),
    iaa.Lambda(lambda images_list, *_: (
        getattr(keras_applications, branch_model_name.lower())
        .preprocess_input(np.stack(images_list), data_format='channels_last')
    )),
])

siamese_nets = SiameseNets(
    branch_model={
        'name': branch_model_name,
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
branch_depth = len(siamese_nets.get_layer('branch_model').layers)

#%% Pre-train branch_model as usual classifier on big classes
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
train_sequence = DeterministicSequence(
    branch_model_train_set,
    preprocessings=preprocessing,
    batch_size=32,
)
classes = train_sequence.targets.columns
val_sequence = DeterministicSequence(
    branch_model_val_set,
    preprocessings=preprocessing,
    batch_size=32,
    classes=classes,
)

branch_classifier = Sequential([
    siamese_nets.get_layer('branch_model'),
    GlobalAveragePooling2D(),
    Dense(len(branch_model_train_set.label.unique()), activation='softmax'),
])
siamese_nets.get_layer('branch_model').trainable = False
optimizer = Adam(lr=1e-4)
branch_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy')
branch_classifier.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=10,
    use_multiprocessing=True,
    workers=5,
)

siamese_nets.get_layer('branch_model').trainable = True
for layer in siamese_nets.get_layer('branch_model').layers[:int(branch_depth * 0.5)]:
    layer.trainable = False
optimizer = Adam(lr=2e-6)
branch_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy')
branch_classifier.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=10,
    epochs=20,
    use_multiprocessing=True,
    workers=5,
)

# %% Check learnt classes
y = branch_classifier.predict_generator(
    DeterministicSequence(
        test_set.loc[lambda df: df.label.isin(branch_model_train_set.label.unique())],
        batch_size=32,
        classes=train_sequence.targets.columns,
        preprocessings=preprocessing,
    ),
    verbose=1,
)

confusion_matrix = (
    test_set.loc[lambda df: df.label.isin(classes)]
    .assign(label_predicted=classes[np.argmax(y, axis=1)])
    .pivot_table(
        index='label_predicted',
        columns='label',
        values='image_name',
        aggfunc='count',
        margins=True,
        fill_value=0,
    )
)

#%% Train model
callbacks = [
    TensorBoard(output_path),
    ModelCheckpoint(
        str(output_path / 'best_model.h5'),
        save_best_only=True,
    ),
    ReduceLROnPlateau(),
]
random_balanced_train_sequence = RandomBalancedPairsSequence(train_set, preprocessings=preprocessing, batch_size=16)
balanced_train_sequence = BalancedPairsSequence(
    train_set, pairs_per_query=2, preprocessings=preprocessing, batch_size=32,
)
random_balanced_val_sequence = RandomBalancedPairsSequence(val_set, preprocessings=preprocessing, batch_size=16)

siamese_nets.get_layer('branch_model').trainable = False
optimizer = Adam(lr=1e-4)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    random_balanced_train_sequence,
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=0,
    epochs=3,
    use_multiprocessing=True,
    workers=5,
)

siamese_nets.get_layer('branch_model').trainable = True
for layer in siamese_nets.get_layer('branch_model').layers[:int(branch_depth * 0.8)]:
    layer.trainable = False
optimizer = Adam(1e-5)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    random_balanced_train_sequence,
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=3,
    epochs=10,
    use_multiprocessing=True,
    workers=5,
)
siamese_nets = load_model(output_path / 'best_model.h5')

siamese_nets.fit_generator(
    balanced_train_sequence,
    steps_per_epoch=(
        len(random_balanced_train_sequence) *
        random_balanced_train_sequence.batch_size / balanced_train_sequence.batch_size
    ),
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=10,
    epochs=15,
    use_multiprocessing=True,
    workers=5,
)
siamese_nets = load_model(output_path / 'best_model.h5')

balanced_train_sequence.batch_size *= 2
balanced_train_sequence.pairs_per_query *= 2
balanced_train_sequence.on_epoch_end()
siamese_nets.fit_generator(
    balanced_train_sequence,
    steps_per_epoch=(
        len(random_balanced_train_sequence) *
        random_balanced_train_sequence.batch_size / balanced_train_sequence.batch_size
    ),
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=15,
    epochs=20,
    use_multiprocessing=True,
    workers=5,
)
siamese_nets = load_model(output_path / 'best_model.h5')

for layer in siamese_nets.get_layer('branch_model').layers[int(branch_depth * 0.5):]:
    layer.trainable = True
optimizer = Adam(1e-6)
siamese_nets.compile(optimizer=optimizer, loss='binary_crossentropy')
siamese_nets.fit_generator(
    random_balanced_train_sequence,
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=20,
    epochs=25,
    use_multiprocessing=True,
    workers=5,
)
siamese_nets = load_model(output_path / 'best_model.h5')

balanced_train_sequence.batch_size /= 2
balanced_train_sequence.pairs_per_query /= 2
balanced_train_sequence.on_epoch_end()
siamese_nets.fit_generator(
    balanced_train_sequence,
    steps_per_epoch=(
        len(random_balanced_train_sequence) *
        random_balanced_train_sequence.batch_size / balanced_train_sequence.batch_size
    ),
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=25,
    epochs=30,
    use_multiprocessing=True,
    workers=5,
)
siamese_nets = load_model(output_path / 'best_model.h5')

balanced_train_sequence.batch_size *= 2
balanced_train_sequence.pairs_per_query *= 2
balanced_train_sequence.on_epoch_end()
siamese_nets.fit_generator(
    balanced_train_sequence,
    steps_per_epoch=(
        len(random_balanced_train_sequence) *
        random_balanced_train_sequence.batch_size / balanced_train_sequence.batch_size
    ),
    validation_data=random_balanced_val_sequence,
    callbacks=callbacks,
    initial_epoch=25,
    epochs=30,
    use_multiprocessing=True,
    workers=5,
)
siamese_nets = load_model(output_path / 'best_model.h5')

#%% Eval on test set
k_shot = 3
n_way = 10
n_episode = 100
test_sequence = DeterministicSequence(test_set, preprocessings=preprocessing, batch_size=16)
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
