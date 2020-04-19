# flake8: noqa: E265
from pathlib import Path

import imgaug.augmenters as iaa
import pandas as pd
from imgaug import parameters as iap
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizer_v2.adam import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

from keras_fsl.dataframe.operators.random_assignment import RandomAssignment
from keras_fsl.models import SiameseDetector
from keras_fsl.sequences import RandomBalancedPairsSequence

#%% Init data
aliases = {
    "ASSIETTE_26": "ASSIETTE_26",
    "ASSIETTE_26_GRIS": "ASSIETTE_26",
    "ASSIETTE_26_NOIR": "ASSIETTE_26",
    "ASSIETTE_BOL_NOIR": "ASSIETTE_26",
    "ASSIETTE_CREUSE": "ASSIETTE_26",
    "ASSIETTE_CREUSE_25_BLANC": "ASSIETTE_26",
    "ASSIETTE_CREUSE_25_BLANC_SEINEWAY": "ASSIETTE_26",
    "ASSIETTE_CREUSE_25_BLEU": "ASSIETTE_26",
    "ASSIETTE_CREUSE_25_GRIS": "ASSIETTE_26",
    "ASSIETTE_CREUSE_25_ROUGE": "ASSIETTE_26",
}
all_annotations = (
    pd.read_csv("data/annotations/all_annotations.csv")
    .assign(
        tray_name=lambda df: df.image_name.str.slice(0, -6),
        image_name=lambda df: "/home/pielectronique/images/" + df.image_name,
        label=lambda df: df.container,
        crop_coordinates=None,
    )
    .drop("container", axis=1)
    .replace(aliases)
    .loc[lambda df: df.label == "ASSIETTE_26"]
    .pipe(RandomAssignment("tray_name"))
)

train_set = all_annotations.loc[lambda df: df.random_split == "train"]
support_set = train_set.assign(crop_coordinates=lambda df: df[["x1", "y1", "x2", "y2"]].agg(list, axis=1))
val_set = all_annotations.loc[lambda df: df.random_split == "val"]

#%% Init training
query_preprocessing = iaa.Sequential(
    [
        iaa.Resize({"longer-side": 416, "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(416, 416),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.MultiplyHueAndSaturation(mul_hue=iap.Uniform(0, 2), mul_saturation=iap.Uniform(1 / 1.5, 1.5)),
        iaa.AssertShape((None, 416, 416, 3)),
    ]
)

support_preprocessing = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-180, 180)),
        iaa.Resize({"longer-side": 128, "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(128, 128, pad_mode="symmetric"),
        iaa.MultiplyHueAndSaturation(mul_hue=iap.Uniform(0, 2), mul_saturation=iap.Uniform(1 / 1.5, 1.5)),
        iaa.AssertShape((None, 128, 128, 3)),
    ]
)

train_sequence = RandomBalancedPairsSequence(
    [train_set, support_set], preprocessing=[query_preprocessing, support_preprocessing], batch_size=16,
)
val_sequence = RandomBalancedPairsSequence(
    [val_set, support_set], preprocessing=[query_preprocessing, support_preprocessing], batch_size=16,
)
output_path = Path("logs") / "siamese_detector" / "assiette_26" / "add_support_in_sequence"
output_path.mkdir(parents=True, exist_ok=True)
siamese_nets = SiameseDetector(
    branch_model={"name": "MobileNet", "init": {"include_top": False, "input_shape": (224, 224, 3)}},
    head_model="DenseSigmoid",
    query_input_shape=(416, 416, 3),
    support_input_shape=(128, 128, 3),
    pooling_layer="GlobalAveragePooling2D",
)

callbacks = [
    TensorBoard(output_path),
    ModelCheckpoint(str(output_path / "best_model.h5"), save_best_only=True),
    ReduceLROnPlateau(),
]

#%% Train the model
siamese_nets.get_layer("branch_model").trainable = False
optimizer = Adam(lr=1e-4)
siamese_nets.compile(optimizer=optimizer, loss="binary_crossentropy")
siamese_nets.fit_generator(
    train_sequence, validation_data=val_sequence, callbacks=callbacks, epochs=2, use_multiprocessing=True, workers=10,
)

siamese_nets.get_layer("branch_model").trainable = True
branch_depth = len(siamese_nets.get_layer("branch_model").layers)
for layer in siamese_nets.get_layer("branch_model").layers[: int(branch_depth * 0.8)]:
    layer.trainable = False

optimizer = Adam(1e-5)
siamese_nets.compile(optimizer=optimizer, loss="binary_crossentropy")
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=2,
    epochs=10,
    use_multiprocessing=True,
    workers=10,
)

for layer in siamese_nets.get_layer("branch_model").layers[int(branch_depth * 0.5) :]:
    layer.trainable = False

optimizer = Adam(1e-5)
siamese_nets.compile(optimizer=optimizer, loss="binary_crossentropy")
siamese_nets.fit_generator(
    train_sequence,
    validation_data=val_sequence,
    callbacks=callbacks,
    initial_epoch=10,
    epochs=15,
    use_multiprocessing=True,
    workers=10,
)

#%% Test model
test_set = all_annotations.loc[lambda df: df.random_split == "test"]
test_preprocessing = iaa.Sequential(
    [
        iaa.Resize({"longer-side": 416, "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(416, 416, position="center"),
        iaa.AssertShape((None, 416, 416, 3)),
    ]
)

tray = img_to_array(
    load_img("/home/pielectronique/images/20151127115553_009901_091_0000000007_B____________1031_001_C.jpg")
).astype(pd.np.uint8)

tray_aug = test_preprocessing.augment_image(tray)

save_img("tray_aug.jpg", tray_aug)

test_sequence = RandomBalancedPairsSequence(
    [test_set, support_set], preprocessing=[test_preprocessing, support_preprocessing], batch_size=16,
)
test_loss = siamese_nets.evaluate_generator(test_sequence)

predictions = siamese_nets.predict_generator(test_sequence, verbose=1, steps=1)
