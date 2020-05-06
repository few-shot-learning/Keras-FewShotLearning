# Welcome to keras-fsl!

As years go by, Few Shot Learning (FSL) and especially Metric Learning is becoming a hot topic not only in academic
papers but also in production applications.

While a lot of researcher nowadays tend to publish their code on github, there is still no easy framework to get
started with FSL. Especially when it comes to benchmarking existing models on personal datasets it is not always easy
to find its path into each single repo. Not mentioning the Tensorflow/PyTorch issue.

This repo aims at filling this gap by providing a single entry-point for Few Shot Learning. It is deeply inspired by
Keras because it shares the same philosophy:

> It was developed with a focus on enabling fast experimentation.
> Being able to go from idea to result with the least possible delay is key to doing good research.

Thus this repo mainly relies on two of the main high-level python packages for data science: Keras and Pandas. While
Pandas may not seem very useful for researchers working with static dataset, it becomes a strong backbone in production
applications when you always need to tinker with your data.

## Few-Shot Learning

Few-shot learning is a task consisting in classifying unseen samples into _n_ classes (so called n way task) where each
classes is only described with few (from 1 to 5 in usual benchmarks) examples.

Most of the state-of-the-art algorithms
try to sort of learn a metric into a well suited (optimized) feature space. Thus deep networks usually first encode the
base images into a feature space onto which a _distance_ or _similarity_ is learnt.

This similarity is meant to be used to later classify samples according to their relative distance, either in a pair-wise
manner where the nearest support set samples is used to classify the query sample ([Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram))
or in a more advanced classifier. Indeed, this philosophy is most commonly known as [the kernel trick](https://en.wikipedia.org/wiki/Kernel_method)
where the kernel is actually the similarity learnt during training. Hence any kind of usual kernel based Machine Learning
could potentially be plugged onto this learnt similarity (see the [min_eigenvalue](keras_fsl/metrics/gram_matrix_metrics.py) metric
to track eigenvalues of the learnt similarity to see if it as actually a kernel).

There is no easy answer to the optimal choice of such a classifier in the feature space. This may depend on performance
as well as on complexity and real application parameters. For instance if the support set is strongly imbalanced, you
may not want to fit an advanced classifier onto it but rather use a raw nearest neighbor approach.

All these considerations lead to the need of a code architecture that will let you play with these parameters with your
own data in order to take the best from them.

Amongst other, the original Siamese Nets is usually known as the network from [Koch et al.](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
This algorithm learns a pair-wise similarity between images. More precisely it uses a densely connected layers on top
of the absolute difference between the two embeddings to predict 0 (different) or 1 (same).

Actually, and as it is now expressed in recent papers, the [representation learning framework](https://arxiv.org/pdf/2002.05709.pdf) is as follows:
 - a data augmentation module A
 - an encoder network E
 - a projection network P
 - a loss L

This repo mimics this framework by proving model builders and notebooks to implement current SOTA algorithms and your
own tweaks seamlessly:
 - use `tf.data.Dataset.map` to apply data augmentation
 - define a `tf.Keras.Sequential` model for your encoder
 - define a `kernel`, ie a `tf.keras.Layer` with two inputs and a real-valued output (see [head models](keras_fsl/models/head_models))
 - use any [`support_layers`](keras_fsl/models/layers/support_layer.py) to wrap the kernel and compute similarities in
 a `tf.keras.Sequential` manner (see notebooks for instance).
 - use any loss chosen accordingly to the output of the `tf.keras.Sequential` model ([GramMatrix](keras_fsl/models/layers/gram_matrix.py) or
 [CentroidsMatrix](keras_fsl/models/layers/centroids_matrix.py) for instance)
 
As an example, the TripletLoss algorithm uses indeed:
 - data augmentation: whatever you want
 - encoder: any backbone like ResNet50 or MobileNet
 - kernel: the l2 norm: `k(x, x') = ||x - x'||^2 = tf.keras.layers.Lambda(lambda inputs: tf.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=1))`
 - support_layer: triplet loss uses all the pair-wises distances, hence it is the GramMatrix
 - loss: `k(a, p) + margin - k(a, n)` with semi-hard mining (see [triplet_loss](keras_fsl/losses/gram_matrix_losses.py))

 
## Overview

This repos provides several tools for few-shot learning:

 - Keras layers and models
 - Keras sequences and Tensorflow datasets for training the models
 - Notebooks with proven learning sequences
 
All these tools can be used all together or separately. One may want to stick with the keras model trained on regular
numpy arrays, with or without callbacks. When designing more advanced `keras.Sequence` or `tf.data.Dataset` for
training, it is advised (and some examples are provided) to use Pandas though it is not necessary at all.

Feel free to experiment and share your thought on this repo by contributing to it! 

## Getting started

The [notebooks](notebooks) section provides some examples. For instance, just run:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential

from keras_fsl.models.encoders import BasicCNN
from keras_fsl.models.layers import GramMatrix
from keras_fsl.losses.gram_matrix_losses import binary_crossentropy
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy, min_eigenvalue
from keras_fsl.utils.tensors import get_dummies


#%% Get data
def preprocessing(input_tensor):
    return tf.cast(input_tensor, tf.float32) / 255


train_dataset, val_dataset, test_dataset = [
    dataset.shuffle(1024).batch(64).map(lambda x, y: (preprocessing(x), get_dummies(y)[0]))
    for dataset in tfds.load(name="omniglot", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True)
]
input_shape = next(tfds.as_numpy(train_dataset.take(1)))[0].shape[1:]  # first shape is batch_size

#%% Training
encoder = BasicCNN(input_shape=input_shape)
support_layer = GramMatrix(kernel="DenseSigmoid")
model = Sequential([encoder, support_layer])
model.compile(optimizer="Adam", loss=binary_crossentropy(), metrics=[classification_accuracy(), min_eigenvalue])
model.fit(train_dataset, validation_data=val_dataset, epochs=5)
```
