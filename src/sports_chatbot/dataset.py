from abc import ABC, abstractmethod

import pandas as pd


class DatasetModule(ABC):
    def __init__(self) -> None:
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        if isinstance(self._data, pd.DataFrame):
            return self._data
        else:
            return self.initialize_dataframe()

    @abstractmethod
    def initialize_dataframe(self) -> pd.DataFrame:
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
