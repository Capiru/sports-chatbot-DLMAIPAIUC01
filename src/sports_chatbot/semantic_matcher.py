import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from sports_chatbot.dataset import DatasetModule


class SemanticMatcherModule:
    def __init__(
        self, dataset_module: DatasetModule, embedder_model: str = "bert-base-uncased"
    ):
        self.dataset_module = dataset_module
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_model)
        self.embedding_model = AutoModel.from_pretrained(embedder_model)
        self.dataset_module.embed_dataframe(self._embed_data)

    def _embed_data(self, data) -> np.ndarray:
        """Embed each line in the CSV file as a vector."""
        lines = data.apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        ).tolist()
        embeddings = []
        for line in lines:
            encoded_input = self.tokenizer(
                line, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)
            # Mean pooling to get a single vector for the sentence
            sentence_embedding = (
                model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
            )
            embeddings.append(sentence_embedding)
        return np.array(embeddings)

    def _find_similar(self, query, top_n=5):
        """Find the most similar lines to the query."""
        encoded_input = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            query_embedding = (
                self.embedding_model(**encoded_input)
                .last_hidden_state.mean(dim=1)
                .squeeze()
                .numpy()
            )
        similarities = cosine_similarity(
            [query_embedding], self.dataset_module.embeddings
        )[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return self.dataset_module.data.iloc[top_indices]

    def get_contextual_response(self, query, top_n=1):
        """Get a response to the query using the most similar lines from the CSV."""
        similar_lines = self._find_similar(query, top_n)
        context = similar_lines.to_string(index=False)
        response = f"Based on the data, here are some relevant information:\n{context}"
        return response
