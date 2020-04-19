#%%
import logging
from pathlib import Path
from unittest.mock import patch

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.saving import load_model

from keras_fsl.datasets import omniglot
from keras_fsl.models import SiameseNets
from keras_fsl.sequences import (
    DeterministicSequence,
    ProtoNetsSequence,
)
from keras_fsl.utils import patch_len, default_workers

# prevent issue with multiprocessing and long sequences, see https://github.com/keras-team/keras/issues/13226
patch_fit_generator = patch(
    "tensorflow.keras.Model.fit_generator", side_effect=default_workers(patch_len(Model.fit_generator))
)
patch_fit_generator.start()
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#%% Get data
train_set, test_set = omniglot.load_data()

#%% Update label columns to be able to mix alphabet during training
train_set = train_set.assign(label=lambda df: df.alphabet + "_" + df.label)
test_set = test_set.assign(label=lambda df: df.alphabet + "_" + df.label)

#%% Training ProtoNets
k_shot = 5
n_way = 5
proto_nets = SiameseNets(
    branch_model="VinyalsNet", head_model={"name": "ProtoNets", "init": {"k_shot": k_shot, "n_way": n_way}}
)
val_set = train_set.sample(frac=0.3, replace=False)
train_set = train_set.loc[lambda df: ~df.index.isin(val_set.index)]
callbacks = [TensorBoard(), ModelCheckpoint("logs/proto_nets/best_weights.h5")]
(Path("logs") / "proto_nets").mkdir(parents=True, exist_ok=True)
preprocessing = iaa.Sequential(
    [iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-10, 10), shear=(-0.8, 1.2))]
)
train_sequence = ProtoNetsSequence(train_set, n_way=n_way, preprocessing=preprocessing, batch_size=16, target_size=(28, 28, 3))
val_sequence = ProtoNetsSequence(val_set, batch_size=16, target_size=(28, 28, 3))

proto_nets.compile(optimizer="Adam", loss="categorical_crossentropy")
Model.fit_generator(  # to use patched fit_generator, see first cell
    proto_nets,
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=100,
    steps_per_epoch=1000,
    validation_steps=200,
    use_multiprocessing=True,
)

#%% Prediction
proto_nets = load_model("logs/proto_nets/best_weights.h5")
encoder = proto_nets.get_layer("branch_model")
head_model = proto_nets.get_layer("head_model")
test_sequence = DeterministicSequence(test_set, batch_size=16, target_size=(28, 28, 3))
embeddings = encoder.predict_generator(test_sequence, verbose=1)

k_shot = 5
n_way = 5
support = (
    test_set.loc[lambda df: df.label.isin(df.label.drop_duplicates().sample(n_way))]
    .groupby("label")
    .apply(lambda group: group.sample(k_shot).drop("label", axis=1))
    .reset_index("label")
)
query = test_set.loc[lambda df: df.label.isin(support.label.unique())].loc[lambda df: ~df.index.isin(support.index)]
predictions = pd.concat(
    [
        query,
        pd.DataFrame(
            head_model.predict(
                [
                    embeddings[query.index],
                    *np.moveaxis(
                        embeddings[np.tile(support.index, reps=len(query))].reshape((len(query.index), k_shot * n_way, -1)),
                        1,
                        0,
                    ),
                ]
            ),
            columns=support.label.unique(),
            index=query.index,
        ),
    ],
    axis=1,
)
confusion_matrix = pd.crosstab(predictions.label, predictions.iloc[:, -n_way:].idxmax(axis=1), margins=True)
