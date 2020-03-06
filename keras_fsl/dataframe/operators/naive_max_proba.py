import numpy as np


class NaiveMaxProba:
    """
    Assign to each label the probability that it reaches the max score amongst all other labels. Note that
    probabilities do not sum up to one since distributions are discretized and consequently two labels can reach maximum
    with non null probability
    """

    def __init__(self, **histogram_kwargs):
        """
        Args:
            **histogram_kwargs: any kwargs to pass to np.histogram
        """
        self.histogram_kwargs = {"bins": 25, "density": True, "range": (0, 1), **histogram_kwargs}

    @staticmethod
    def get_p(dataframe):
        """
        Compute for each label the probability that it reaches the maximum score:
        .. math: P[max_i X_i = X_j]

        Distributions are assumed to be independent (naive assumption).
        """
        probabilities = []
        for i, row in dataframe.iterrows():
            distributions = dataframe.cdf.copy()
            distributions[i] = row.pdf.copy()
            probabilities += [np.sum(np.prod(np.stack(distributions.tolist(), axis=0), axis=0))]
        return probabilities

    def __call__(self, input_dataframe):
        """
        Args:
            input_dataframe (pandas.DataFrame): with column image_name, label and score

        Returns:
            pandas.DataFrame: dataframe with one score per image per label
        """
        return (
            input_dataframe.groupby(["image_name", "label"], as_index=False)
            .agg({"score": list})
            .assign(
                pdf=lambda df: (
                    df.score.apply(lambda values: np.histogram(values, **self.histogram_kwargs)[0])
                    / self.histogram_kwargs["bins"]
                ),
                cdf=lambda df: df.pdf.apply(lambda p: np.cumsum(p)),
            )
            .groupby("image_name")
            .apply(lambda group: group.assign(confidence=lambda df: self.get_p(df)))
            .reset_index(drop=True)
            .drop(["pdf", "cdf"], axis=1)
        )
