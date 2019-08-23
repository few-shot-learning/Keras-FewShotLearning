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
            raise ValueError('pairs_per_query should be even')
        self.pairs_per_query = pairs_per_query
        self._support_labels = None
        super().__init__(*args, **kwargs)
        if self.batch_size % 2 == 1:
            raise ValueError(f'batch_size should be even')

    @property
    def support_labels(self):
        if self._support_labels is None:
            self._support_labels = self.support_annotations.label.value_counts()
        return self._support_labels

    def on_epoch_end(self):
        self.query_samples = (
            self.query_annotations
            .sample(frac=1)
        )
        self.support_samples = (
            pd.concat(
                self.query_samples
                .apply(lambda row: (self.get_batch_for_sample(row.label)), axis=1)
                .values,
            )
            .reset_index(drop=True)
        )
        self.query_samples = (
            self.query_samples
            .iloc[pd.np.repeat(self.query_annotations.index, self.pairs_per_query)]
            .reset_index(drop=True)
        )

    def get_batch_for_sample(self, anchor_label):
        replace = self.support_labels[anchor_label] < self.pairs_per_query
        positive_samples = (
            self.support_annotations
            .loc[lambda df: df.label == anchor_label]
            .sample(self.pairs_per_query // 2, replace=replace)
        )
        negative_samples = (
            self.support_annotations
            .loc[lambda df: df.label != anchor_label]
            .sample(self.pairs_per_query // 2)
        )
        return pd.concat([positive_samples, negative_samples])

    def __len__(self):
        return math.ceil(len(self.query_annotations) * self.pairs_per_query / self.batch_size)
