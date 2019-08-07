import abc

import pandas as pd
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array


class AbstractPairsSequence(Sequence, metaclass=abc.ABCMeta):
    """
    Base class for pairs sequences. All initialization should
    """

    def __init__(self, annotations, batch_size, **load_img_kwargs):
        """
        Args:
            annotations (Union[pandas.DataFrame, list[pandas.DataFrame]]): query and support annotations. If a single
            dataframe is given, it will be used both for query and support set.
            batch_size (int): number of images per batch
        """
        if not isinstance(annotations, list):
            annotations = [annotations]
        self.query_annotations = annotations[0]
        self.query_samples = pd.DataFrame()
        self.support_annotations = annotations[-1]
        self.support_samples = pd.DataFrame()
        self.load_img_kwargs = load_img_kwargs
        self.batch_size = batch_size
        self.on_epoch_end()

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return [
            pd.np.stack(
               self.query_samples
               .iloc[start_index:end_index]
               .apply(lambda row: img_to_array(load_img(row.image_name, **self.load_img_kwargs)))
            ),
            pd.np.stack(
               self.support_samples
               .iloc[start_index:end_index]
               .apply(lambda row: img_to_array(load_img(row.image_name, **self.load_img_kwargs)))
            ),
        ], self.targets[start_index:end_index]

    @property
    def targets(self):
        return self.query_samples.label == self.support_samples.label

    def __len__(self):
        return len(self.query_annotations) / self.batch_size
