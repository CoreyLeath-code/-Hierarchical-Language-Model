import re

from .config import ModelConfig


class HierarchicalTokenizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.word_to_id = {word: idx for idx, word in enumerate(self.config.vocab)}
        self.pad_id = self.word_to_id.get("<PAD>", 0)
        self.unk_id = self.word_to_id.get("<UNK>", 1)

    def _tokenize_sentence(self, sentence: str) -> list[int]:
        # Clean string and isolate words
        words = re.findall(r"\w+", sentence.lower())
        tokens = [self.word_to_id.get(w, self.unk_id) for w in words]

        # Truncate or Pad to exact max_seq_len contract boundary
        if len(tokens) > self.config.max_seq_len:
            return tokens[: self.config.max_seq_len]
        return tokens + [self.pad_id] * (self.config.max_seq_len - len(tokens))

    def encode_document(self, document: str) -> list[list[int]]:
        # Split document into sentences using basic punctuation boundaries
        sentences = [s.strip() for s in re.split(r"[.!?]", document) if s.strip()]

        # Encode individual sentences
        encoded_doc = [self._tokenize_sentence(s) for s in sentences]

        # Truncate or Pad sentence count to exact max_sentences boundary
        empty_sentence = [self.pad_id] * self.config.max_seq_len
        if len(encoded_doc) > self.config.max_sentences:
            return encoded_doc[: self.config.max_sentences]
        return encoded_doc + [empty_sentence] * (self.config.max_sentences - len(encoded_doc))
