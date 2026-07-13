import torch
import torch.nn as nn


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected x input shape: [Batch * Sentences, Sequence_Length]
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)

        # Mean pooling across tokens to compress words into a unified vector
        sentence_vector = torch.mean(outputs, dim=1)
        return torch.tanh(self.fc(sentence_vector))


class HierarchicalLM(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, token_hidden: int, doc_hidden: int, num_classes: int
    ):
        super().__init__()
        self.token_encoder = TokenEncoder(vocab_size, embed_dim, token_hidden)
        self.doc_gru = nn.GRU(token_hidden, doc_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(doc_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Target 3D Tensor shape: [Batch_Size, Num_Sentences, Sequence_Length]
        batch_size, num_sentences, seq_len = x.size()

        # Flatten structure into 2D to enable massive parallel processing across token slices
        flat_x = x.view(batch_size * num_sentences, seq_len)
        sentence_vectors = self.token_encoder(flat_x)

        # Reconstruct structural document dimensionality
        doc_inputs = sentence_vectors.view(batch_size, num_sentences, -1)

        # Track hierarchical dependencies across sentences
        doc_outputs, _ = self.doc_gru(doc_inputs)

        # Mean pooling across sentence contexts to get a unified document fingerprint
        doc_vector = torch.mean(doc_outputs, dim=1)
        return self.classifier(doc_vector)
