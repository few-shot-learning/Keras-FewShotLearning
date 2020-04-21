import numpy as np
import pandas as pd

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class AbstractPairsSequence(AbstractSequence):
    """
    Base class for pairs sequences. All initialization should
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            annotations (Union[pandas.DataFrame, list[pandas.DataFrame]]): query and support annotations. If a single
            dataframe is given, it will be used both for query and support set.
            batch_size (int): number of images per batch
        """
        super().__init__(*args, **kwargs)
        self.query_preprocessing = self.preprocessings[0]
        self.support_preprocessing = self.preprocessings[-1]
        self.query_annotations = self.annotations[0]
        self.support_annotations = self.annotations[-1]
        self.support_annotations_by_label = {group[0]: group[1] for group in self.support_annotations.groupby("label")}

        self.query_samples = pd.DataFrame()
        self.support_samples = pd.DataFrame()

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        query_images = self.query_preprocessing(images=self.load_img(self.query_samples.iloc[start_index:end_index]))
        support_images = self.support_preprocessing(images=self.load_img(self.support_samples.iloc[start_index:end_index]))
        return [np.stack(query_images), np.stack(support_images)], self.targets[start_index:end_index].astype(int)

    @property
    def targets(self):
        return self.query_samples.label == self.support_samples.label
