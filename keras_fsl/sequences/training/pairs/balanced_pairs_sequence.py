import math

import pandas as pd

from .abstract_pairs_sequence import AbstractPairsSequence


class BalancedPairsSequence(AbstractPairsSequence):
    """
    Iterate over all the query set and uses each image with half matching and half
    non-matching pairs.
    """

    def __init__(self, *args, pairs_per_query=2, **kwargs):
        """

        Args:
            *args:
            pairs_per_query (int): number of pairs done with each query image. This should be an even number since each
                query image is used in as many matching as non-matching pairs
            **kwargs:
        """
        if pairs_per_query % 2 == 1:
            raise ValueError("pairs_per_query should be even")
        self.pairs_per_query = pairs_per_query
        if kwargs["batch_size"] % 2 == 1:
            raise ValueError(f"batch_size should be even")

        super().__init__(*args, **kwargs)
        self.support_labels = self.support_annotations.label.value_counts()
        self.on_epoch_end()

    def on_epoch_end(self):
        indexes = self.query_annotations.index.values
        pd.np.random.shuffle(indexes)
        self.query_samples = self.query_annotations.loc[indexes]
        self.support_samples = pd.concat(self.query_samples.label.map(self.get_batch_for_sample).tolist()).reset_index(
            drop=True
        )
        self.query_samples = self.query_samples.loc[lambda df: df.index.repeat(self.pairs_per_query)].reset_index(drop=True)

    def get_batch_for_sample(self, anchor_label):
        replace = self.support_labels[anchor_label] < self.pairs_per_query
        positive_samples = self.support_annotations.loc[lambda df: df.label == anchor_label].sample(
            self.pairs_per_query // 2, replace=replace
        )
        negative_samples = self.support_annotations.loc[lambda df: df.label != anchor_label].sample(self.pairs_per_query // 2)
        return pd.concat([positive_samples, negative_samples])

    def __len__(self):
        return math.ceil(len(self.query_annotations) * self.pairs_per_query / self.batch_size)
