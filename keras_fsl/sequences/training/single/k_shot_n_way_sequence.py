import math
import warnings
import pandas as pd

from keras_fsl.sequences.training.single.deterministic_sequence import DeterministicSequence


class KShotNWaySequence(DeterministicSequence):
    """
    Generate k-shots n-ways batch of data. Note that batch_size is consequently overridden with k*n
    """

    def __init__(
        self, annotations, batch_size, k_shot, n_way, **kwargs,
    ):
        """
        Args:
            *args: args to be passed to super constructor
            k_shot (int): Number of samples of same class in each batch.
            n_way (int): Number of difference classes in each batch.
            **kwargs: kwargs to be passed to super constructor
        """
        if batch_size is not None and batch_size != n_way * k_shot:
            warnings.warn("batch_size was set but not consistent with k_shot and n_way. Value is overridden.")
            batch_size = k_shot * n_way
        if kwargs.get("shuffle"):
            warnings.warn("KShotNWaySequence does not use the shuffle attribute")
            kwargs["shuffle"] = False

        self.k_shot = k_shot
        self.n_way = n_way
        super().__init__(annotations, batch_size, **kwargs)
        self.label_to_indexes = dict(
            self.annotations[0].reset_index().groupby("label", as_index=False).agg({"index": list})[["label", "index"]].values
        )

    def on_epoch_end(self):
        self.annotations[0] = (
            self.annotations[0]
            .groupby("label")
            .apply(
                lambda group: (
                    group.sample(frac=1).assign(
                        k_shot_index=[
                            group.name + "-" + str(index)
                            for index in pd.np.repeat(list(range(math.ceil(len(group) / self.k_shot))), self.k_shot)
                        ][: len(group)]
                    )
                )
            )
            .reset_index("label", drop=True)
            .groupby("k_shot_index")
            .apply(lambda group: (group.assign(k_shot_len=len(group))))
        )
        indexes_with_k_shots = pd.np.array(
            pd.np.random.permutation(
                self.annotations[0]
                .reset_index()
                .loc[lambda df: df.k_shot_len == self.k_shot]
                .groupby("k_shot_index", as_index=False)
                .agg({"index": list})["index"]
                .values
            ).tolist()
        ).flatten()
        other_indexes = self.annotations[0].index.difference(indexes_with_k_shots)
        indexes = indexes_with_k_shots.tolist() + other_indexes.tolist()
        self.annotations[0] = self.annotations[0].loc[indexes]
        self.targets = self.targets.loc[indexes]

    def __len__(self):
        return math.floor(len(self.annotations[0]) / self.batch_size)
