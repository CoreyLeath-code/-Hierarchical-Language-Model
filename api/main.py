from fastapi import FastAPI

from api.inference import ENABLE_LIVE_MODEL, MODEL_NAME, MODEL_REVISION, llm_engine
from api.schemas import PromptRequest, PromptResponse

app = FastAPI(
    title="Hierarchical Language Model API",
    description="Research-safe API for hierarchical language model generation.",
    version="1.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "live_model_enabled": ENABLE_LIVE_MODEL,
    }


@app.post("/generate", response_model=PromptResponse)
def generate_text(request: PromptRequest):
    output = llm_engine.generate(request.prompt, max_new_tokens=request.max_tokens)
    return {"output": output}
