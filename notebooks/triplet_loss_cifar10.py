"""
This notebooks borrows from https://www.tensorflow.org/addons/tutorials/losses_triplet and is intended to compare tf.addons triplet loss
implementation against this one. It is also aimed at benchmarking the impact of the distance function.
"""
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Sequential

from keras_fsl.layers import GramMatrix
from keras_fsl.losses.gram_matrix_losses import binary_crossentropy, triplet_loss
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy
from keras_fsl.utils.tensors import get_dummies


#%% Build datasets
def preprocessing(input_tensor):
    return tf.cast(input_tensor, tf.float32) / 255


train_dataset, val_dataset, test_dataset = tfds.load(
    name="cifar10", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True
)
train_dataset = train_dataset.shuffle(1024).batch(64, drop_remainder=True)
val_dataset = val_dataset.shuffle(1024).batch(64, drop_remainder=True)
test_dataset = test_dataset.batch(64, drop_remainder=True)

train_labels = [batch[1].numpy().tolist() for batch in train_dataset]
val_labels = [batch[1].numpy().tolist() for batch in val_dataset]
test_labels = [batch[1].numpy().tolist() for batch in test_dataset]
train_steps = len(train_labels)
val_steps = len(val_labels)
test_steps = len(test_labels)

input_shape = next(train_dataset.take(1).as_numpy_iterator())[0].shape

print(
    pd.concat(
        [
            pd.DataFrame({"label": tf.nest.flatten(train_labels)}).assign(split="train"),
            pd.DataFrame({"label": tf.nest.flatten(val_labels)}).assign(split="val"),
            pd.DataFrame({"label": tf.nest.flatten(test_labels)}).assign(split="test"),
        ]
    )
    .groupby("split")
    .apply(lambda group: pd.get_dummies(group.label).agg("sum"))
)

results = []

#%% Save test labels for later visualization in projector https://projector.tensorflow.org/
out_m = io.open("meta.tsv", "w", encoding="utf-8")
for img, labels in tfds.as_numpy(test_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()

#%% Build model
encoder = Sequential(
    [
        Input(input_shape[1:]),
        Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        GlobalMaxPooling2D(),
    ]
)
encoder.save_weights("initial_encoder.h5")

#%% Train encoder with usual cross entropy
encoder.load_weights("initial_encoder.h5")
classifier = Sequential([encoder, Dense(10, activation="softmax")])
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]
)
classifier.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(),
    epochs=50,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(
        lambda x, y: (preprocessing(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard("sparse_categorical_crossentropy")],
)
loss, accuracy = classifier.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), y)), steps=test_steps)
results += [{"name": "classifier", "loss": loss, "accuracy": accuracy}]
embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), y)), steps=test_steps)
np.savetxt("classifier_embeddings.tsv", embeddings, delimiter="\t")

#%% Train with triplet loss
norms = {
    "l2": lambda inputs: tf.math.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=1),
    "l1": lambda inputs: tf.math.reduce_sum(tf.abs(inputs[0] - inputs[1]), axis=1),
    "cosine_similarity": lambda inputs: 1 - cosine_similarity(inputs[0], inputs[1], axis=1),
    "softmax_abs": lambda x: tf.nn.softmax(tf.math.abs(x[0] - x[1])),
}
for name, norm in norms.items():
    encoder.load_weights("initial_encoder.h5")
    model = Sequential([encoder, GramMatrix(kernel=Lambda(norm))])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(), metrics=[classification_accuracy(ascending=True)],
    )
    model.fit(
        train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
        epochs=50,
        steps_per_epoch=train_steps,
        validation_data=val_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
        validation_steps=val_steps,
        callbacks=[TensorBoard(f"{name}_triplet_loss")],
    )
    results += [
        {
            "name": name,
            **dict(
                zip(
                    model.metrics_names,
                    model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps),
                )
            ),
        }
    ]
    embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)
    np.savetxt(f"{name}_embeddings.tsv", embeddings, delimiter="\t")

#%% Try with mixed norm
encoder.load_weights("initial_encoder.h5")
model = Sequential(
    [encoder, GramMatrix(kernel={"name": "MixedNorms", "init": {"activation": "relu", "norms": list(norms.values())}})]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(), metrics=[classification_accuracy(ascending=True)],
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    epochs=50,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard("mixed_triplet_loss")],
)
results += [
    {
        "name": "mixed_norms",
        **dict(
            zip(
                model.metrics_names,
                model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps),
            )
        ),
    }
]

embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)
np.savetxt(f"mixed_norms_embeddings.tsv", embeddings, delimiter="\t")

#%% Try with mixed similarity
encoder.load_weights("initial_encoder.h5")
model = Sequential(
    [encoder, GramMatrix(kernel={"name": "MixedNorms", "init": {"activation": "sigmoid", "norms": list(norms.values())}})]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=binary_crossentropy(), metrics=[classification_accuracy(ascending=False)],
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    epochs=50,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard("mixed_triplet_loss")],
)
results += [
    {
        "name": "mixed_similarity",
        **dict(
            zip(
                model.metrics_names,
                model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps),
            )
        ),
    }
]
embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)
np.savetxt(f"mixed_similarity_embeddings.tsv", embeddings, delimiter="\t")

#%% Try with learnt norm
encoder.load_weights("initial_encoder.h5")
model = Sequential([encoder, GramMatrix(kernel={"name": "LearntNorms", "init": {"activation": "relu"}})])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(), metrics=[classification_accuracy(ascending=True)],
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    epochs=100,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard("learnt_triplet_loss")],
)
results += [
    {
        "name": "learnt_norms",
        **dict(
            zip(
                model.metrics_names,
                model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps),
            )
        ),
    }
]
embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)
np.savetxt(f"learnt_norms_embeddings.tsv", embeddings, delimiter="\t")

#%% Try with learnt similarity
encoder.load_weights("initial_encoder.h5")
model = Sequential([encoder, GramMatrix(kernel={"name": "LearntNorms", "init": {"activation": "sigmoid"}})])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=binary_crossentropy(), metrics=[classification_accuracy(ascending=True)],
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    epochs=100,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard("learnt_triplet_loss")],
)
results += [
    {
        "name": "learnt_similarity",
        **dict(
            zip(
                model.metrics_names,
                model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps),
            )
        ),
    }
]
embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)
np.savetxt(f"learnt_similarity_embeddings.tsv", embeddings, delimiter="\t")

#%% Export final stats
pd.concat(results).to_csv("results.csv", index=False)
