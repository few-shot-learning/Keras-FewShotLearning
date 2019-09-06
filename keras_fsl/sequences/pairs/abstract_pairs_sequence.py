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
        self.query_samples = pd.DataFrame()
        self.support_samples = pd.DataFrame()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return [
                   self.load_img(self.query_samples.iloc[start_index:end_index]),
                   self.load_img(self.support_samples.iloc[start_index:end_index]),
               ], self.targets[start_index:end_index]

    @property
    def targets(self):
        return self.query_samples.label == self.support_samples.label
