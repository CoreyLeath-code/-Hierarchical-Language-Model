from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=100, ge=1, le=1024)


class PromptResponse(BaseModel):
    output: str
