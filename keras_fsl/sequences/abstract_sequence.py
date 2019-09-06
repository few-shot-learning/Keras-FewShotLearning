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
        self.preprocessing = preprocessing if preprocessing is not None else iaa.Sequential()
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.query_annotations) / self.batch_size)

    @property
    def support_labels(self):
        if self._support_labels is None:
            self._support_labels = self.support_annotations.label.value_counts()
        return self._support_labels

    def load_img(self, input_dataframe):
        return self.preprocessing.augment_images(pd.np.stack(
            input_dataframe
            .apply(
                lambda row: img_to_array(load_img(row.image_name, **self.load_img_kwargs)),
                axis=1,
            )
        ))
