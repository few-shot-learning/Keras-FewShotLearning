"""
Compare training with usual categorical crossentropy and with gram matrix and binary crossentropy on CIFAR-10
We use a simple CNN in order to challenge only the training setting
"""
# flake8: noqa: E265
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical

from keras_fsl.models.branch_models import BasicCNN
from keras_fsl.losses import binary_crossentropy, mean_score_classification_loss
from keras_fsl.metrics import min_eigenvalue
from keras_fsl.models.layers import GramMatrix, Classification

#%% Init data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

#%% Train with usual categorical crossentropy
model = BasicCNN((32, 32, 3), classes=num_classes)
model.summary()
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
)
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64)

#%% Visualize embeddings
output_dir = "output_dir"

embeddings = tf.Variable(model.predict(X_test))  # tf.Tensor are not trackable
ckpt = tf.train.Checkpoint(embeddings=embeddings)
checkpoint_file = output_dir + "/embeddings.ckpt"
ckpt.save(checkpoint_file)

reader = tf.train.load_checkpoint(output_dir)
map = reader.get_variable_to_shape_map()
key_to_use = ""
for key in map:
    if "embeddings" in key:
        key_to_use = key

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = key_to_use

projector.visualize_embeddings(output_dir, config)

#%% Evaluate model
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Accuracy: {scores[1]:.2%}")

#%% Train with binary crossentropy and gram matrix
accuracies = []
for i in range(1, 21):
    kernel = Lambda(lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1))
    model = Sequential([BasicCNN((32, 32, 3), i), GramMatrix(kernel)])
    model.summary()
    model.compile(
        optimizer="adam", loss=binary_crossentropy(), metrics=[mean_score_classification_loss, min_eigenvalue],
    )
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)

    embeddings = model.layers[0].predict(X_train)
    classifier = Sequential([model.layers[0], Classification(kernel)])
    classifier.layers[1].set_support_set(embeddings, y_train)
    classifier.compile(loss="binary_crossentropy", optimizer="adam")
    classifier.evaluate(X_test, y_test, verbose=1)
    y_pred = classifier.predict(X_test, verbose=1)
    confusion_matrix = pd.crosstab(
        index=pd.Categorical(np.argmax(y_pred, axis=1), categories=list(range(10))),
        columns=pd.Categorical(np.argmax(y_test, axis=1), categories=list(range(10))),
        margins=True,
        dropna=False,
        rownames=["pred"],
        colnames=["true"],
    )
    accuracies.append(np.diag(confusion_matrix)[:-1].sum() / np.diag(confusion_matrix)[-1])
