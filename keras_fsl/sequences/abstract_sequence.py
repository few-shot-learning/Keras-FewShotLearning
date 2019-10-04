import math
from abc import ABCMeta

import pandas as pd
import imgaug.augmenters as iaa
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.utils import Sequence


class AbstractSequence(Sequence, metaclass=ABCMeta):

    def __init__(self, annotations, batch_size, preprocessing=None, **load_img_kwargs):
        """
        Args:
            annotations (Union[pandas.DataFrame, list[pandas.DataFrame]]): query and support annotations. If a single
            dataframe is given, it will be used both for query and support set.
            batch_size (int): number of images per batch
            preprocessing (imgaug.augmenters.meta.Augmenter): augmenter for data augmentation
        """
        if not isinstance(annotations, list):
            annotations = [annotations]
        self.query_annotations = annotations[0]
        self.support_annotations = annotations[-1]
        self.support_annotations_by_label = {
            group[0]: group[1]
            for group in self.support_annotations.groupby('label')
        }
        self.load_img_kwargs = load_img_kwargs
        self.batch_size = batch_size
        self._support_labels = None
        if not isinstance(preprocessing, list):
            preprocessing = [preprocessing]
        self.query_preprocessing = preprocessing[0] if preprocessing[0] is not None else iaa.Sequential()
        self.support_preprocessing = preprocessing[-1] if preprocessing[-1] is not None else iaa.Sequential()
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.query_annotations) / self.batch_size)

    @property
    def support_labels(self):
        if self._support_labels is None:
            self._support_labels = self.support_annotations.label.value_counts()
        return self._support_labels

    def load_query_img(self, input_dataframe):
        return pd.np.stack(
            input_dataframe
            .apply(lambda row: (
                self.query_preprocessing.augment_image(img_to_array(load_img(row.image_name, **self.load_img_kwargs)))
            ), axis=1)
        )

    def load_support_img(self, input_dataframe):
        return pd.np.stack(
            input_dataframe
            .apply(lambda row: (
                self.support_preprocessing.augment_image(img_to_array(load_img(row.image_name, **self.load_img_kwargs)))
            ), axis=1)
        )
