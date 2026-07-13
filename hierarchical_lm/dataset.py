import torch
from torch.utils.data import Dataset

from .config import ModelConfig
from .tokenizer import HierarchicalTokenizer


class HierarchicalTextDataset(Dataset):
    def __init__(self, documents: list[str], labels: list[int], config: ModelConfig):
        if len(documents) != len(labels):
            raise ValueError("documents and labels must contain the same number of items")

        self.labels = labels
        self.config = config
        self.tokenizer = HierarchicalTokenizer(config)

        # Pre-compile data arrays directly to tensors
        self.data_tensors = [
            torch.tensor(self.tokenizer.encode_document(doc), dtype=torch.long) for doc in documents
        ]

    def __len__(self) -> int:
        return len(self.data_tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data_tensors[idx], torch.tensor(self.labels[idx], dtype=torch.long)
