import pytest

from hierarchical_lm.config import ModelConfig
from hierarchical_lm.dataset import HierarchicalTextDataset
from hierarchical_lm.tokenizer import HierarchicalTokenizer


def test_model_config_requires_positive_dimensions():
    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(vocab_size=0)


def test_model_config_requires_special_tokens():
    with pytest.raises(ValueError, match="<PAD> and <UNK>"):
        ModelConfig(vocab=["the", "system"])


def test_dataset_rejects_label_mismatch():
    config = ModelConfig()

    with pytest.raises(ValueError, match="same number"):
        HierarchicalTextDataset(["one document"], [], config)


def test_tokenizer_preserves_tensor_contract_dimensions():
    config = ModelConfig(max_sentences=3, max_seq_len=4)
    tokenizer = HierarchicalTokenizer(config)

    encoded = tokenizer.encode_document("The system is scalable. Traffic is dense.")

    assert len(encoded) == 3
    assert all(len(sentence) == 4 for sentence in encoded)
