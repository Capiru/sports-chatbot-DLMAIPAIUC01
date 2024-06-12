import pandas as pd

from sports_chatbot.dataset import CSVDataset


def test_dataset_module():
    dataset_module = CSVDataset("./data/sports_chatbot/Day_1.csv")
    assert len(dataset_module.data) > 0
    assert isinstance(dataset_module.data, pd.DataFrame)
