"""Inference adapter for live and CI-safe text generation."""

import os

import torch

MODEL_NAME = os.getenv("HLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
MODEL_REVISION = os.getenv("HLM_MODEL_REVISION", "main")
ENABLE_LIVE_MODEL = os.getenv("HLM_ENABLE_LIVE_MODEL", "false").lower() == "true"


class LLMEngine:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_revision = MODEL_REVISION
        self.enable_live_model = ENABLE_LIVE_MODEL
        self.tokenizer = None
        self.model = None

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer  # pragma: no cover

        self.tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover
            self.model_name,
            revision=self.model_revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(  # pragma: no cover
            self.model_name,
            revision=self.model_revision,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        if not prompt or not prompt.strip():
            return "Prompt cannot be empty."

        if not self.enable_live_model:
            return (
                "Live model loading is disabled. Set HLM_ENABLE_LIVE_MODEL=true "
                "and configure HLM_MODEL_NAME to enable generation."
            )

        self._load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


llm_engine = LLMEngine()
