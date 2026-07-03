import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
# Import your core tokenizer package assets
from hierarchical_lm.config import ModelConfig
from hierarchical_lm.tokenizer import HierarchicalTokenizer

app = FastAPI(
    title="Hierarchical Language Model Inference Core",
    description="Enterprise Ingress Layer serving dual-stage document analysis engines Engine Sub-System."
)

# Initialize configuration and parsing components at runtime startup
config = ModelConfig()
tokenizer = HierarchicalTokenizer(config)

class DocumentSubmission(BaseModel):
    text: str = Field(..., example="The execution pipeline is online. Automation matrix reports absolute green.")

@app.post("/api/v1/inference", tags=["Production Ingress"])
async def process_document_inference(payload: DocumentSubmission):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Data Contract Exception: Input text body cannot be null.")
    
    try:
        # Tier 4 execution: Slice text structure cleanly into an aligned integer array
        tokenized_matrix = tokenizer.encode_document(payload.text)
        
        # Format list arrays into standard multidimensional list arrays matching tensor requirements
        # In a multi-node deployment, this array is serialized to gRPC or dispatched to an inference worker pool
        input_tensor = torch.tensor([tokenized_matrix], dtype=torch.long)
        
        return {
            "status": "Ingested",
            "dimensions": list(input_tensor.shape),
            "payload_preview": payload.text[:60] + "..."
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Core Processing Engine Exception: {str(err)}")

@app.get("/healthz", tags=["System Telemetry"])
async def system_health_check():
    return {"status": "Healthy", "tier_integrity": "Green"}
