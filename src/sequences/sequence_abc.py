import abc

from keras.utils import Sequence


class AbstractSequence(Sequence, metaclass=abc.ABCMeta):
    """A base class inheriting from keras Sequence to generate batches of images for the models"""

    def __init__(self, annotations, batch_size, model=None):
        """
        Annotations should be a list of dataframes.
        Args:
            annotations (list[pandas.DataFrame]): list of standard annotations dataframes
            batch_size (int): number of images per batch
            model (keras.models.Model): model the sequence will be used by
        """
        self.annotations = annotations
        self.batch_size = batch_size
        self.model = model

    def config(self):
        return {'annotations': self.annotations, 'batch_size': self.batch_size, 'model': self.model}
