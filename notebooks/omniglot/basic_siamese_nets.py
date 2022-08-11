# flake8: noqa: E265
import os
from pathlib import Path

import imgaug.augmenters as iaa
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizer_v2.adam import Adam
from keras.saving import load_model

from keras_fsl.datasets import omniglot
from keras_fsl.models import SiameseNets
from keras_fsl.sequences import (
    DeterministicSequence,
    RandomBalancedPairsSequence,
)

#%% Get data
train_set, test_set = omniglot.load_data()

#%% Update label columns to be able to mix alphabet during training
train_set = train_set.assign(label=lambda df: df.alphabet + "_" + df.label)
test_set = test_set.assign(label=lambda df: df.alphabet + "_" + df.label)

#%% Training basic SiameseNets
output_path = Path("logs") / "koch_nets"
output_path.mkdir(parents=True, exist_ok=True)
siamese_nets = SiameseNets()
val_set = train_set.sample(frac=0.3, replace=False)
train_set = train_set.loc[lambda df: ~df.index.isin(val_set.index)]
callbacks = [
    TensorBoard(output_path),
    ModelCheckpoint(str(output_path / "best_model.h5"), save_best_only=True),
    ReduceLROnPlateau(verbose=1),
    EarlyStopping(patience=10),
]
preprocessing = iaa.Sequential(
    [iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-10, 10), shear=(-0.8, 1.2))]
)
train_sequence = RandomBalancedPairsSequence(train_set, preprocessing=preprocessing, batch_size=128)
val_sequence = RandomBalancedPairsSequence(val_set, preprocessing=preprocessing, batch_size=128)

optimizer = Adam(1e-4)
siamese_nets.compile(optimizer=optimizer, loss="binary_crossentropy")
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    epochs=150,
    use_multiprocessing=True,
    workers=os.cpu_count(),
)

#%% Prediction
siamese_nets = load_model(output_path / "best_model.h5")
encoder = siamese_nets.get_layer("branch_model")
head_model = siamese_nets.get_layer("head_model")
test_sequence = DeterministicSequence(test_set, batch_size=16)
embeddings = encoder.predict_generator(test_sequence, verbose=1)

k_shot = 1
n_way = 5
support = (
    test_set.loc[lambda df: df.label.isin(test_set.label.drop_duplicates().sample(n_way))]
    .groupby("label")
    .apply(lambda group: group.sample(k_shot).drop("label", axis=1))
    .reset_index("label")
)
query = (
    test_set.loc[lambda df: df.label.isin(support.label.unique())]
    .loc[lambda df: ~df.index.isin(support.index)]
    .loc[lambda df: df.index.repeat(k_shot * n_way)]
    .reset_index()
)
support = support.loc[lambda df: pd.np.tile(df.index, len(query) // (k_shot * n_way))].reset_index()
predictions = pd.concat(
    [
        query,
        pd.DataFrame(head_model.predict([embeddings[query["index"]], embeddings[support["index"]]]), columns=["score"]),
        support.add_suffix("_support"),
    ],
    axis=1,
)
confusion_matrix = (
    predictions.groupby(query.columns.to_list())
    .apply(lambda group: group.nlargest(1, columns="score").label_support)
    .reset_index()
    .pivot_table(values="image_name", index="label_support", columns="label", aggfunc="count", margins=True, fill_value=0)
    .assign(precision=lambda df: pd.np.diag(df)[:-1] / df.All[:-1])
    .T.assign(recall=lambda df: pd.np.diag(df)[:-1] / df.All[:-2])
    .T.assign(f1=lambda df: 2 * df.precision * df.loc["recall"] / (df.precision + df.loc["recall"]))
)
