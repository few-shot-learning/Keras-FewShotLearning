"""
This notebooks borrows from https://www.tensorflow.org/addons/tutorials/losses_triplet and is intended to compare tf.addons triplet loss
implementation against this one. It is also aimed at benchmarking the impact of the distance function.
"""
import io

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential

from keras_fsl.losses.gram_matrix_losses import triplet_loss
from keras_fsl.models.layers import GramMatrix
from keras_fsl.utils.tensors import get_dummies

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
support_layer = GramMatrix(kernel=Lambda(lambda inputs: tf.math.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=1)))
model = Sequential([encoder, support_layer])
encoder.save_weights("initial_encoder.h5")


#%% Build datasets
def preprocessing(input_tensor):
    return tf.cast(input_tensor, tf.float32) / 255


train_dataset, test_dataset = tfds.load(name="mnist", split=["train", "test"], as_supervised=True)
train_dataset = train_dataset.shuffle(1024).batch(32)
test_dataset = test_dataset.shuffle(1024).batch(32)

#%% Train with tfa triplet loss
encoder.load_weights("initial_encoder.h5")
encoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss())
encoder.fit(train_dataset.map(lambda x, y: (preprocessing(x), y)), epochs=5)
encoder.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), y)))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), y)))
np.savetxt("tfa_embeddings.tsv", results, delimiter="\t")

#%% Train with keras_fsl triplet loss
encoder.load_weights("initial_encoder.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss())
model.fit(train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), epochs=5)
model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
results = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])))
np.savetxt("keras_fsl_embeddings.tsv", results, delimiter="\t")

#%% Save test embeddings for visualization in projector https://projector.tensorflow.org/
out_m = io.open("meta.tsv", "w", encoding="utf-8")
for img, labels in tfds.as_numpy(test_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()
