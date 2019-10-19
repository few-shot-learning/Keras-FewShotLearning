from abc import ABC, abstractmethod


class AbstractOperator(ABC):
    """
    Abstract class for all dataframe operators to be used with piping
    """

    @abstractmethod
    def __call__(self, input_dataframe):
        pass
