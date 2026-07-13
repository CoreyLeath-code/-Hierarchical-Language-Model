from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 5000
    embed_dim: int = 128
    token_hidden: int = 256
    doc_hidden: int = 256
    num_classes: int = 2
    max_sentences: int = 10
    max_seq_len: int = 20
    learning_rate: float = 0.001

    # Simple mock vocabulary for demonstration boundaries
    vocab: list[str] = field(
        default_factory=lambda: [
            "<PAD>",
            "<UNK>",
            "the",
            "system",
            "is",
            "scalable",
            "and",
            "highly",
            "efficient",
            "traffic",
            "dense",
        ]
    )

    def __post_init__(self) -> None:
        numeric_fields = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "token_hidden": self.token_hidden,
            "doc_hidden": self.doc_hidden,
            "num_classes": self.num_classes,
            "max_sentences": self.max_sentences,
            "max_seq_len": self.max_seq_len,
            "learning_rate": self.learning_rate,
        }
        invalid = [name for name, value in numeric_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"ModelConfig fields must be positive: {', '.join(invalid)}")

        if "<PAD>" not in self.vocab or "<UNK>" not in self.vocab:
            raise ValueError("ModelConfig vocabulary must include <PAD> and <UNK> tokens")
