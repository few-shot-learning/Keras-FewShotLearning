# Welcome to Keras-FewShotLearning!

As years come and go, Few Shot Learning (FSL) is becoming a hot topic not only in academic papers but also in production
applications.

While a lot of researcher nowadays tend to publish their code on github, there is still no easy and framework to get
started with FSL. Especially when it comes to benchmarking existing models on personal datasets it is not always easy
to find its path into each single repo. Not mentioning the Tensorflow/PyTorch issue.

This repo aims to fill this gap by providing a single entry-point for Few Shot Learning. It is deeply inspired by Keras
because it shares the same philosophy:

> It was developed with a focus on enabling fast experimentation.
> Being able to go from idea to result with the least possible delay is key to doing good research.

Thus this repo mainly relies on two of the main high-level python packages for data science: Keras and Pandas. While
Pandas may not seem very useful for researchers working with static dataset, it becomes a strong backbone in production
applications when you always need to tinker with your data.

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
model = SiameseNets(
    branch_model={'name': 'SingleConv2D', 'init': {'input_shape': (105, 105, 3)}},
    head_model='DenseSigmoid',
)
train_sequence = RandomPairsSequence(train_set, batch_size=16)
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit_generator(train_sequence)
```
