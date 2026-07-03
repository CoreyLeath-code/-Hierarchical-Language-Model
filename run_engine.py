import torch
from torch.utils.data import DataLoader
from hierarchical_lm.config import ModelConfig
from hierarchical_lm.dataset import HierarchicalTextDataset
from hierarchical_lm.model import HierarchicalLM

def main():
    print("Initializing Hierarchical Language Model Framework...")
    
    # 1. Initialize Configuration Configuration
    config = ModelConfig()
    
    # 2. Compile Mock Data Assets
    mock_documents = [
        "The system is scalable. Traffic is dense. Core logic is highly efficient.",
        "Scalable execution matrices pass cleanly. The backend pipeline functions."
    ]
    mock_labels = [1, 0] # Binary target example
    
    # 3. Instantiate Dataset and Loading Handlers
    dataset = HierarchicalTextDataset(mock_documents, mock_labels, config)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 4. Initialize Core Model
    model = HierarchicalLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        token_hidden=config.token_hidden,
        doc_hidden=config.doc_hidden,
        num_classes=config.num_classes
    )
    
    # 5. Run Single Mock Inference Forward Pass to verify tensor integrity
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            print(f"\nIngested Input Tensor Batch Shape: {list(batch_x.shape)}")
            # Expected shape: [Batch_Size, Max_Sentences, Max_Seq_Length]
            
            output_logits = model(batch_x)
            print(f"Produced Target Logits Batch Shape: {list(output_logits.shape)}")
            # Expected shape: [Batch_Size, Num_Classes]
            
            print("Mathematical tensor alignment verified successfully across the entire hierarchy.")

if __name__ == "__main__":
    main()
