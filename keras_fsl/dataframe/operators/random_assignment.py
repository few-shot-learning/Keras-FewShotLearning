import numpy as np


class RandomAssignment:
    """
    Group the input dataframe and assign to each group a random value with np.random.choice into the column_name arg.
    """

    def __init__(self, by, choices=None, p=None, column_name="random_split"):
        self.by = by
        self.choices = choices if choices is not None else ["train", "val", "test"]
        self.p = p if p is not None else [0.7, 0.1, 0.2]
        self.column_name = column_name

    def __call__(self, input_dataframe):
        return (
            input_dataframe.groupby(self.by)
            .apply(lambda group: (group.assign(random_split_tmp=np.random.choice(self.choices, p=self.p))))
            .rename(columns={"random_split_tmp": self.column_name})
            .reset_index(drop=True)
        )
