# Welcome to keras-fsl!

As years go by, Few Shot Learning (FSL) is becoming a hot topic not only in academic papers but also in production
applications.

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
where the kernel is indeed the similarity learnt during training. Hence any kind of usual kernel based Machine Learning
could potentially be plugged onto this learnt similarity.

There is no easy answer to the optimal choice of such a classifier in the feature space. This may depend on performance
as well as one complexity and real application parameters. For instance if the support set is strongly imbalanced, you
may not want to fit an advanced classifier onto it but rather use a raw nearest neighbor approach.

All these considerations lead to the need of a code architecture that will let you play with these parameters with your
own data in order to take the best from them.

Amongst other, the Siamese Nets is usually known as the network from [Koch et al.](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
This algorithm learns a pair-wise similarity between images. More precisely it uses a densely connected layers on top
of the difference between the two embeddings to predict 0 (different) or 1 (same).

In this repo we have called Siamese nets all the algorithms built within the same framework, i.e. choosing a backbone
and a _similarity_ to evaluate the embeddings. In this context the well known [Protypical networks](https://arxiv.org/pdf/1703.05175.pdf)
falls into the Siamese Nets frameworks and is available here as `SiameseNets(head_model="ProtoNets")`.

## Overview

This repos provides several tools for few-shot learning:

 - Keras layers and models
 - Keras sequences and Tensorflow datasets for training the models
 - Notebooks with proven learning sequences
 
All these tools can be used all together or separately. One may want to stick with the keras model trained on regular
numpy arrays, with or without callbacks. When designing more advanced `keras.Sequence` or `tf.data.Dataset` for
training, it is advised (and some examples are provided) to use Pandas though it is not necessary at all.

We think that fast experimentation requires the good level of modularity. Modularity means that the flow of operations
should be described as a sequence of operations with well defined interfaces. Furthermore you should be able to change
or update any of these single operations without changing anything else. For these reasons we think that stacked Layers
as well as stacked (chained) Pandas or tf.data operations are the right way of building ML pipelines. In this context we
also rely on ImgAug for preprocessing the data.

Feel free to experiment and share your thought on this repo by contributing to it! 

## Getting started

The [notebooks](notebooks) section provides some examples. For instance, just run:

```python
from keras_fsl.models import SiameseNets
from keras_fsl.datasets import omniglot
from keras_fsl.sequences.training.pairs import RandomPairsSequence

# %% Get data
train_set, test_set = omniglot.load_data()

# %% Update label columns to be able to mix alphabet during training
train_set = train_set.assign(label=lambda df: df.alphabet + '_' + df.label)
test_set = test_set.assign(label=lambda df: df.alphabet + '_' + df.label)

# %% Training
model = SiameseNets()
train_sequence = RandomPairsSequence(train_set, batch_size=16)
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit_generator(train_sequence)
```
