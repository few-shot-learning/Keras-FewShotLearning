#%%
from pathlib import Path
from unittest.mock import patch

import imgaug.augmenters as iaa
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import Model

from keras_fsl.datasets import omniglot
from keras_fsl.models import SiameseNets
from keras_fsl.sequences import (
    DeterministicSequence,
    RandomBalancedPairsSequence,
)
from keras_fsl.utils import patch_len, default_workers
# prevent issue with multiprocessing and long sequences, see https://github.com/keras-team/keras/issues/13226
patch_fit_generator = patch(
    'tensorflow.keras.Model.fit_generator',
    side_effect=default_workers(patch_len(Model.fit_generator)),
)
patch_fit_generator.start()

#%% Get data
train_set, test_set = omniglot.load_data()

#%% Update label columns to be able to mix alphabet during training
train_set = train_set.assign(label=lambda df: df.alphabet + '_' + df.label)
test_set = test_set.assign(label=lambda df: df.alphabet + '_' + df.label)

#%% Training basic SiameseNets
siamese_nets = SiameseNets()
val_set = train_set.sample(frac=0.3, replace=False)
train_set = train_set.loc[lambda df: ~df.index.isin(val_set.index)]
callbacks = [TensorBoard(), ModelCheckpoint('logs/proto_nets/best_weights.h5')]
(Path('logs') / 'proto_nets').mkdir(parents=True, exist_ok=True)
preprocessing = iaa.Sequential([
    iaa.Affine(
        translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
        rotate=(-10, 10),
        shear=(-0.8, 1.2),
    )
])
train_sequence = RandomBalancedPairsSequence(train_set, preprocessing=preprocessing, batch_size=16)
val_sequence = RandomBalancedPairsSequence(val_set, batch_size=16)

siamese_nets.compile(optimizer='Adam', loss='binary_crossentropy')
Model.fit_generator(  # to use patched fit_generator, see first cell
    siamese_nets,
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=100,
    steps_per_epoch=1000,
    validation_steps=200,
    use_multiprocessing=True,
)

#%% Prediction
encoder = siamese_nets.get_layer('branch_model')
head_model = siamese_nets.get_layer('head_model')
test_sequence = DeterministicSequence(test_set, batch_size=16)
embeddings = encoder.predict_generator(test_sequence, verbose=1)

k_shot = 1
n_way = 5
support = (
    test_set
    .loc[lambda df: df.label.isin(test_set.label.drop_duplicates().sample(n_way))]
    .groupby('label')
    .apply(lambda group: group.sample(k_shot).drop('label', axis=1))
    .reset_index('label')
)
query = (
    test_set
    .loc[lambda df: df.label.isin(support.label.unique())]
    .loc[lambda df: ~df.index.isin(support.index)]
    .loc[lambda df: df.index.repeat(k_shot * n_way)]
    .reset_index()
)
support = (
    support
    .loc[lambda df: pd.np.tile(df.index, len(query) // (k_shot * n_way))]
    .reset_index()
)
predictions = (
    pd.concat([
        query,
        pd.DataFrame(head_model.predict([embeddings[query['index']], embeddings[support['index']]]), columns=['score']),
        support.add_suffix('_support'),
    ], axis=1)
)
confusion_matrix = (
    predictions
    .groupby(query.columns.to_list())
    .apply(lambda group: group.nlargest(1, columns='score').label_support)
    .reset_index()
    .pivot_table(
        values='image_name',
        index='label_support',
        columns='label',
        aggfunc='count',
        margins=True,
        fill_value=0,
    )
    .assign(precision=lambda df: pd.np.diag(df)[:-1] / df.All[:-1])
    .T.assign(recall=lambda df: pd.np.diag(df)[:-1] / df.All[:-2]).T
    .assign(f1=lambda df: 2 * df.precision * df.loc['recall'] / (df.precision + df.loc['recall']))
)

