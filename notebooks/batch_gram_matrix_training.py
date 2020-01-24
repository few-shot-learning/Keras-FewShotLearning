#%%
from datetime import datetime
from pathlib import Path

import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
import yaml
from tensorflow.keras import applications as keras_applications
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keras_fsl.models import SiameseNets
from keras_fsl.models.layers import Classification, GramMatrix
from keras_fsl.sequences import prediction, training
from keras_fsl.losses import binary_crossentropy, accuracy, mean_score_classification_loss, min_eigenvalue
# tf.config.experimental_run_functions_eagerly(True)

def build_k_way_n_shot_dataset(annotations, n_shot, k_way, classes=None, to_categorical=True, training=True):
    annotations = annotations.assign(label=pd.Categorical(annotations.label, categories=classes))

    targets = annotations.label.cat.codes
    if to_categorical:
        targets = (
            pd.get_dummies(targets)
              .reindex(list(range(len(targets.unique()))), axis=1)
              .fillna(0)
        )

    num_classes = len(targets.columns)
    batch_size = n_shot * k_way

    def load_image_and_crop(image_path, x1, y1, x2, y2):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image)
        image = (tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2
        image = tf.image.crop_to_bounding_box(image, y1, x1, y2 - y1, x2 - x1)
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)

        return image

    def data_aug(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        return image

    def build_datasets_for_class(annotations, targets, index_class):
        print(f"Building for {index_class}")
        class_targets = targets[targets[index_class] > 0]
        class_annotations = annotations.loc[class_targets.index]

        dataset = tf.data.Dataset.from_tensor_slices((
            class_annotations["image_name"],
            class_annotations["x1"],
            class_annotations["y1"],
            class_annotations["x2"],
            class_annotations["y2"],
            class_targets.values.astype("float32"),
        ))

        dataset = dataset.map(
            lambda image_name, x1, y1, x2, y2, target: (load_image_and_crop(image_name, x1, y1, x2, y2), target),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if training:
            dataset = dataset.cache()
            dataset = dataset.map(
                lambda image, target: (data_aug(image), target),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        return dataset

    datasets_by_class = [build_datasets_for_class(annotations, targets, index_class=index_class) for index_class in targets.columns]

    choice_dataset = tf.data.Dataset.range(num_classes).shuffle(buffer_size=num_classes).repeat().interleave(
        lambda index: tf.data.Dataset.from_tensors(index).repeat(n_shot),
        cycle_length=1,
        block_length=n_shot,
    )

    dataset = tf.data.experimental.choose_from_datasets(datasets_by_class, choice_dataset).batch(batch_size)

    return dataset


#%% Init data
output_folder = Path('logs') / 'kernel_loss' / datetime.today().strftime('%Y%m%d-%H%M%S')
output_folder.mkdir(parents=True, exist_ok=True)
try:
    shutil.copy(__file__, output_folder / 'training_pipeline.py')
except (FileNotFoundError, NameError):
    pass

all_annotations = (
    pd.read_csv('data/annotations/cropped_images.csv')
    .assign(
        day=lambda df: df.image_name.str.slice(3, 11),
        image_name=lambda df: 'data/images/cropped_images/' + df.image_name,
    )
)
train_val_test_split = yaml.safe_load(open('data/annotations/cropped_images_split.yaml'))
train_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['train_set_dates'])]
val_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['val_set_dates'])]
test_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split['test_set_dates'])].reset_index(drop=True)

#%% Init model
branch_model_name = 'MobileNet'
batch_size = 64
n_shot = batch_size // 8
k_way = 8
num_classes = len(train_set.label.unique())

print("Building image dataset.")
dataset = build_k_way_n_shot_dataset(train_set, k_way=k_way, n_shot=n_shot, training=True).repeat()
print("Done training.")
val_dataset = build_k_way_n_shot_dataset(val_set, k_way=k_way, n_shot=n_shot, training=False).repeat()
print("Done validation.")

siamese_nets = SiameseNets(
    branch_model={
        'name': branch_model_name,
        'init': {'include_top': False, 'input_shape': (224, 224, 3), 'pooling': 'avg'}
    },
    head_model={
        'name': 'MixedNorms',
        'init': {
            'norms': [
                lambda x: 1 - tf.nn.l2_normalize(x[0]) * tf.nn.l2_normalize(x[1]),
                lambda x: tf.math.abs(x[0] - x[1]),
                lambda x: tf.nn.softmax(tf.math.abs(x[0] - x[1])),
                lambda x: tf.square(x[0] - x[1]),
            ],
            'use_bias': False,
        }
    }
)

model = Sequential([
    siamese_nets.get_layer('branch_model'),
    GramMatrix(kernel=siamese_nets.get_layer('head_model')),
])

#%% Init training
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
batch_size = 64
callbacks = [
    TensorBoard(output_folder, write_images=True, histogram_freq=1),
    ModelCheckpoint(
        str(output_folder / 'kernel_loss_best_loss_weights.h5'),
        save_best_only=True,
        save_weights_only=True,
    ),
    ModelCheckpoint(
        str(output_folder / 'kernel_loss_best_accuracy_weights.h5'),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
    ),
    ReduceLROnPlateau(),
]

#%% Train model with loss on kernel
siamese_nets.get_layer('branch_model').trainable = False
optimizer = Adam(lr=1e-4)
margin = 0.05
model.compile(
    optimizer=optimizer,
    loss=binary_crossentropy(margin),
    metrics=[binary_crossentropy(0.0), accuracy(margin), mean_score_classification_loss, min_eigenvalue],
)
model.fit(
    dataset,
    validation_data=val_dataset,
    callbacks=callbacks,
    initial_epoch=0,
    steps_per_epoch=361,
    validation_steps=78,
    epochs=10,
)

siamese_nets.get_layer('branch_model').trainable = True
optimizer = Adam(lr=1e-5)
model.compile(
    optimizer=optimizer,
    loss=binary_crossentropy(margin),
    metrics=[binary_crossentropy(0.0), accuracy(margin), mean_score_classification_loss, min_eigenvalue],
)
model.fit_generator(
    dataset,
    validation_data=val_dataset,
    callbacks=callbacks,
    steps_per_epoch=361,
    validation_steps=78,
    initial_epoch=10,
    epochs=100,
)

model.save(output_folder / 'final_model.h5')

#%% Eval on test set
k_shot = 1
n_way = 5
n_episode = 100
test_sequence = training.single.DeterministicSequence(test_set, preprocessings=preprocessing, batch_size=16)
embeddings = siamese_nets.get_layer('branch_model').predict_generator(test_sequence, verbose=1)

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
    test_sequence = prediction.pairs.ProductSequence(
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
plt.clf()
scores.boxplot()
plt.savefig(output_folder / 'scores_boxplot.png')
plt.clf()
scores.good_prediction.hist()
plt.savefig(output_folder / 'scores_good_predictions.png')
scores.to_csv(output_folder / 'scores.csv', index=False)

#%% Export classification model with SavedModel
model.load_weights(str(output_folder / 'kernel_loss_best_loss_weights.h5'))
classifier = Sequential([
    siamese_nets.get_layer('branch_model'),
    Classification(kernel=siamese_nets.get_layer('head_model')),
    Activation('softmax'),
])
tf.saved_model.save(classifier, str(output_folder / 'saved_model/1/'))

#%% Example of use as classifier
classifier.get_layer('classification').set_support_set(
    support_tensors=tf.convert_to_tensor(support_set_embeddings, tf.float32),
    support_labels=tf.convert_to_tensor(pd.get_dummies(support_set.label.values).values, tf.float32),
)
y = classifier.predict_generator(test_sequence, verbose=1)
