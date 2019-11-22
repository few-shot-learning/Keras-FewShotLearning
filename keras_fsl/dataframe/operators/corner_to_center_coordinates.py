from .abstract_operator import AbstractOperator


class CornerToCenterCoordinates(AbstractOperator):
    """
    Change a dataframe with x1, y1, x2, y2 coordinates into a dataframe with width, height and center position
    """

    def __call__(self, input_dataframe):
        return (
            input_dataframe
            .assign(
                width=lambda df: df.x2 - df.x1,
                height=lambda df: df.y2 - df.y1,
                x=lambda df: (df.x1 + df.x2) // 2,
                y=lambda df: (df.y1 + df.y2) // 2,
            )
        )
