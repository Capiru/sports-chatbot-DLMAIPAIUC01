from abc import ABC

from transformers import AutoModelForQuestionAnswering, pipeline

from sports_chatbot.dataset import CSVDataset
from sports_chatbot.semantic_matcher import SemanticMatcherModule


class ChatbotModule(ABC):
    def __init__(self, semantic_matcher: SemanticMatcherModule) -> None:
        super().__init__()
        self.semantic_matcher = semantic_matcher


class SportsChatbot(ChatbotModule):
    def __init__(self, csv_path, model_name="bert-base-uncased"):
        super().__init__(
            semantic_matcher=SemanticMatcherModule(CSVDataset(csv_path), model_name)
        )
        qa_model = "deepset/roberta-base-squad2"
        self.model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        self.nlp = pipeline("question-answering", model=qa_model, tokenizer=qa_model)

    def get_contextual_response(self, query, top_n=1):
        return self.semantic_matcher.get_contextual_response(query, top_n)

    def query(self, query):
        return self.nlp(question=query, context=self.get_contextual_response(query))
