import math
from abc import ABCMeta

from tensorflow.python.keras.utils import Sequence


class AbstractSequence(Sequence, metaclass=ABCMeta):

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
        self.support_annotations = annotations[-1]
        self.support_annotations_by_label = {
            group[0]: group[1]
            for group in self.support_annotations.groupby('label')
        }
        self.load_img_kwargs = load_img_kwargs
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.query_annotations) / self.batch_size)
