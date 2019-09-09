# Welcome to Keras-FewShotLearning!

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

Amongst other, the Siamese Nets is usually known as the network from [Koch et al.](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
This algorithm learns a pair-wise similarity between images. More precisely it uses a densely connected layers on top
of the difference between the two embeddings to predict 0 (different) or 1 (same).

In this repo we have called Siamese nets all the algorithms built within the same framework, i.e. choosing a backbone
and a _head model_ to evaluate the embeddings. In this context the well known [Protypical networks](https://arxiv.org/pdf/1703.05175.pdf)
falls into the Siamese Nets frameworks and is available here as `SiameseNets(head_model="ProtoNets")`.

## Overview

This repos provides several tools for few-shot learning:

 - Keras models
 - Keras sequences for training the models
 - Keras callbacks to monitor the trainings
 
All these tools can be used all together or separately. One may want to stick with the keras model trained on regular
numpy arrays, with or without callbacks. When designing more advanced `keras.Sequence` for training, it is advised (and
some examples are provided) to use Pandas though it is not necessary at all.

We think that fast experimentation requires the good level of modularity. Modularity means that the flow of operations
should be described as a sequence of operations with well defined interfaces. Furthermore you should be able to change
or update any of these single operations without changing anything else. For these reasons we think that stacked Layers
as well as stacked (chained) Pandas operations are the right way of building ML pipelines. In this context we also rely
on ImgAug for preprocessing the data. Feel free to experiment and share your thought on this repo by contributing to it! 

## Getting started

The [notebooks](notebooks) section provides examples on how to run several models onto the standard FSL dataset. As an
example, just run:

```python
from keras_fsl.models import SiameseNets
from keras_fsl.datasets import omniglot
from keras_fsl.sequences import RandomPairsSequence

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
