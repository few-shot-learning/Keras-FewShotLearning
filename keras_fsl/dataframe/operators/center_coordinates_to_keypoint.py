from imgaug import Keypoint

from .abstract_operator import AbstractOperator


class CenterCoordinatesToKeypoint(AbstractOperator):
    """
    Change a dataframe with x1, y1, x2, y2 coordinates into a dataframe a single imgaug.BoundingBox series
    """

    def __call__(self, input_dataframe):
        return (
            input_dataframe
            .assign(center=lambda df: df[['x', 'y']].apply(lambda row: Keypoint(**row), axis=1))
        )
