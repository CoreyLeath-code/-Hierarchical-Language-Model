from dataclasses import dataclass, field
from typing import List

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
    vocab: List[str] = field(default_factory=lambda: [
        "<PAD>", "<UNK>", "the", "system", "is", "scalable", 
        "and", "highly", "efficient", "traffic", "is", "dense"
    ])
