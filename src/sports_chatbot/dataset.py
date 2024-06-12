from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd


class DatasetModule(ABC):
    def __init__(self) -> None:
        self._data = None
        self._embeddings = None

    @property
    def data(self) -> pd.DataFrame:
        if isinstance(self._data, pd.DataFrame):
            return self._data
        else:
            return self.initialize_dataframe()

    @property
    def embeddings(self) -> np.ndarray:
        if isinstance(self._embeddings, np.ndarray):
            return self._embeddings
        else:
            raise Exception(
                "Please call embed_dataframe with the embedding function before calling for the embeddings!"
            )

    @abstractmethod
    def initialize_dataframe(self) -> pd.DataFrame:
        raise Exception(
            "You should implement the initialize dataframe logic for your dataset!"
        )

    @abstractmethod
    def embed_dataframe(self, embed_fn: Callable) -> np.ndarray:
        raise Exception(
            "You should implement the initialize dataframe logic for your dataset!"
        )


class CSVDataset(DatasetModule):
    def __init__(self, csv_file_path: str) -> None:
        super().__init__()
        self.csv_file_path = csv_file_path

    def initialize_dataframe(self) -> pd.DataFrame:
        self._data = pd.read_csv(self.csv_file_path)
        return self.data

    def embed_dataframe(self, embed_fn: Callable) -> np.ndarray:
        self._embeddings = embed_fn(self.data)
        return self.embeddings
