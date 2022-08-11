"""
This notebooks borrows from https://www.tensorflow.org/addons/tutorials/losses_triplet and is intended to compare tf.addons triplet loss
implementation against this one. It is also aimed at benchmarking the impact of the distance function.
"""
import io

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import TensorBoard

from keras_fsl.losses.gram_matrix_losses import triplet_loss
from keras_fsl.layers import GramMatrix
from keras_fsl.utils.tensors import get_dummies
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy

#%% Build model
encoder = Sequential(
    [
        Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation=None),  # No activation on final dense layer
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),  # L2 normalize embeddings
    ]
)
encoder.save_weights("initial_encoder.h5")
support_layer = GramMatrix(kernel=Lambda(lambda inputs: tf.math.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=1)))
model = Sequential([encoder, support_layer])


#%% Build datasets
def preprocessing(input_tensor):
    return tf.cast(input_tensor, tf.float32) / 255


train_dataset, test_dataset = tfds.load(name="mnist", split=["train", "test"], as_supervised=True)
train_dataset = train_dataset.shuffle(1024).batch(32)
test_dataset = test_dataset.batch(32)

#%% Save test labels for later visualization in projector https://projector.tensorflow.org/
out_m = io.open("meta.tsv", "w", encoding="utf-8")
for img, labels in tfds.as_numpy(test_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()

#%% Train with tfa triplet loss
encoder.load_weights("initial_encoder.h5")
encoder.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss(),
)
encoder.fit(train_dataset.map(lambda x, y: (preprocessing(x), y)), epochs=5, callbacks=[TensorBoard("tfa_loss")])
encoder.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), y)))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), y)))
np.savetxt("tfa_embeddings.tsv", results, delimiter="\t")

#%% Train with keras_fsl triplet loss
encoder.load_weights("initial_encoder.h5")
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(), metrics=[classification_accuracy(ascending=True)]
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), epochs=5, callbacks=[TensorBoard("keras_fsl_loss")]
)
model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
np.savetxt("keras_fsl_embeddings.tsv", results, delimiter="\t")

#%% Try with l1 norm
support_layer.kernel = Lambda(lambda inputs: tf.math.reduce_sum(tf.abs(inputs[0] - inputs[1]), axis=1))
encoder.load_weights("initial_encoder.h5")
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(), metrics=[classification_accuracy(ascending=True)]
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])),
    epochs=5,
    callbacks=[TensorBoard("keras_fsl_l1_loss")],
)
model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
np.savetxt("keras_fsl_l1_embeddings.tsv", results, delimiter="\t")

#%% Try with cosine similarity
support_layer.kernel = Lambda(
    lambda inputs: tf.reduce_sum(1 - tf.nn.l2_normalize(inputs[0], axis=1) * tf.nn.l2_normalize(inputs[1], axis=1), axis=1)
)
encoder.load_weights("initial_encoder.h5")
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(0.1), metrics=[classification_accuracy(ascending=True)]
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])),
    epochs=5,
    callbacks=[TensorBoard("keras_fsl_cosine_similarity")],
)
model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
np.savetxt("keras_fsl_cosine_similarity_embeddings.tsv", results, delimiter="\t")

#%% Try with learnt norm
support_layer = GramMatrix(
    kernel={
        "name": "MixedNorms",
        "init": {"activation": "relu", "norms": [lambda x: tf.math.abs(x[0] - x[1]), lambda x: tf.math.square(x[0] - x[1])]},
    }
)
encoder.load_weights("initial_encoder.h5")
model = Sequential([encoder, support_layer])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(), metrics=[classification_accuracy(ascending=True)]
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])),
    epochs=5,
    callbacks=[TensorBoard("keras_fsl_learnt_norm")],
)
model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
np.savetxt("keras_fsl_learnt_norm_embeddings.tsv", results, delimiter="\t")

#%% Try with learnt similarity
support_layer = GramMatrix(
    kernel={
        "name": "MixedNorms",
        "init": {
            "activation": "sigmoid",
            "norms": [lambda x: tf.math.abs(x[0] - x[1]), lambda x: tf.math.square(x[0] - x[1])],
        },
    }
)
encoder.load_weights("initial_encoder.h5")
model = Sequential([encoder, support_layer])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss(0.1), metrics=[classification_accuracy(ascending=True)]
)
model.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])),
    epochs=5,
    callbacks=[TensorBoard("keras_fsl_learnt_similarity")],
)
model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
np.savetxt("keras_fsl_learnt_similarity_embeddings.tsv", results, delimiter="\t")
