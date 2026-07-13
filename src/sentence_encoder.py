# src/sentence_encoder.py

"""
Sentence Encoder Module
-----------------------
Encodes individual sentences into vector representations using
a pretrained Transformer (e.g., BERT) from Hugging Face.
This forms the first layer of the Hierarchical Language Model (HLM).
"""

import hashlib
import os

import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str | None = None,
        embedding_dim: int = 768,
        live_model: bool | None = None,
    ):
        """
        Initializes the sentence encoder.

        Args:
            model_name (str): Hugging Face model name to load.
            device (str): "cuda" or "cpu" depending on availability.
        """
        super(SentenceEncoder, self).__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.live_model = (
            os.getenv("HLM_ENABLE_LIVE_SENTENCE_MODEL", "false").lower() == "true"
            if live_model is None
            else live_model
        )
        self.tokenizer = None
        self.model = None

        if self.live_model:
            from transformers import AutoModel, AutoTokenizer  # pragma: no cover

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # pragma: no cover
            self.model = AutoModel.from_pretrained(model_name).to(self.device)  # pragma: no cover

    def _deterministic_embedding(self, sentence: str) -> torch.Tensor:
        digest = hashlib.sha256(sentence.encode("utf-8")).digest()
        values = []
        while len(values) < self.embedding_dim:
            values.extend((byte / 255.0) for byte in digest)
            digest = hashlib.sha256(digest).digest()
        return torch.tensor(values[: self.embedding_dim], dtype=torch.float32)

    def forward(self, sentences):
        """
        Encodes a list of sentences into embeddings.

        Args:
            sentences (list of str): Input sentences.

        Returns:
            torch.Tensor: Sentence embeddings [batch_size, hidden_dim].
        """
        if not self.live_model:
            return torch.stack(
                [self._deterministic_embedding(sentence) for sentence in sentences]
            ).to(self.device)

        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )

        # Run through model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Take the [CLS] token embedding as sentence representation
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]

        return sentence_embeddings
