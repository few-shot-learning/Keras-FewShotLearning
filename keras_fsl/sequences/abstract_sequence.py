import math

import imgaug.augmenters as iaa
import numpy as np
from abc import ABCMeta
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence


class AbstractSequence(Sequence, metaclass=ABCMeta):
    def __init__(self, annotations, batch_size, preprocessings=None, **load_img_kwargs):
        """
        Args:
            annotations (Union[pandas.DataFrame, list[pandas.DataFrame]]): query and support annotations. If a single
            dataframe is given, it will be used both for query and support set.
            batch_size (int): number of images per batch
            preprocessings (Union[imgaug.augmenters.meta.Augmenter, List[imgaug.augmenters.meta.Augmenter]]):
                augmenters for data augmentation. There should by either one single augmenter or one augmenter per
                annotation
        """
        if not isinstance(annotations, list):
            annotations = [annotations]
        self.annotations = [
            annotations_.assign(crop_coordinates=lambda df: df.get("crop_coordinates")) for annotations_ in annotations
        ]

        self.load_img_kwargs = load_img_kwargs
        self.batch_size = batch_size
        self._support_labels = None

        if type(preprocessings) is not list:
            preprocessings = [preprocessings]
        self.preprocessings = [
            preprocessing if preprocessing is not None else iaa.Sequential() for preprocessing in preprocessings
        ]

    def __len__(self):
        return math.ceil(len(self.annotations[0]) / self.batch_size)

    def load_img(self, input_dataframe):
        return [
            img_to_array(load_img(image_name, **self.load_img_kwargs).crop(crop_coordinates)).astype(np.uint8)
            for image_name, crop_coordinates in zip(input_dataframe.image_name, input_dataframe.crop_coordinates)
        ]
