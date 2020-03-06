from imgaug import BoundingBox

from .abstract_operator import AbstractOperator


class CoordinatesToBoundingBox(AbstractOperator):
    """
    Change a dataframe with x1, y1, x2, y2 coordinates into a dataframe a single imgaug.BoundingBox series
    """

    def __call__(self, input_dataframe):
        return input_dataframe.assign(
            bounding_box=lambda df: df[["x1", "y1", "x2", "y2"]].apply(lambda row: BoundingBox(**row), axis=1)
        )
