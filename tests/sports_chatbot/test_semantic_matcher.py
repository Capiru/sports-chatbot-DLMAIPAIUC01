from sports_chatbot.dataset import CSVDataset
from sports_chatbot.semantic_matcher import SemanticMatcherModule


def test_semantic_matcher(csv_path):
    semantic_matcher = SemanticMatcherModule(CSVDataset(csv_path))
    # TODO: context matcher is probably returning header as well
    context = semantic_matcher.get_contextual_response("Football", top_n=3)
    assert "real madrid" in context.lower()
    assert "barcelona" in context.lower()
