from pytest import fixture


@fixture
def csv_path() -> str:
    return "./data/sports_chatbot/Day_1.csv"
