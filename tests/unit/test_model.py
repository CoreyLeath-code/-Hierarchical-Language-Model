import unittest
import torch
from hierarchical_lm.config import ModelConfig
from hierarchical_lm.model import HierarchicalLM

class TestHierarchicalModelDimensions(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.model = HierarchicalLM(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            token_hidden=self.config.token_hidden,
            doc_hidden=self.config.doc_hidden,
            num_classes=self.config.num_classes
        )

    def test_forward_pass_dimensions(self):
        # Create a dynamic mock batch [Batch=4, Sentences=10, Tokens=20]
        mock_input = torch.randint(
            0, self.config.vocab_size, 
            (4, self.config.max_sentences, self.config.max_seq_len)
        )
        
        output = self.model(mock_input)
        
        # Enforce that the output dimension matches the class layout contract
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], self.config.num_classes)

if __name__ == "__main__":
    unittest.main()
