"""Exact sentence log-probabilities from a causal LLM (TinyLlama, Llama-3.2, Mistral, GPT-2)."""

from __future__ import annotations

from typing import Optional


def load_lm(name: str, adapter: Optional[str] = None, device_map: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)  # token: set HF_TOKEN in the environment
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return LMScorer(model.eval(), tokenizer)


class LMScorer:
    """log P(text) = sum over tokens of log p(token | BOS, previous tokens)."""

    def __init__(self, model, tokenizer, batch_size: int = 16):
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size
        self.bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.cache: dict[str, float] = {}

    def _ids(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [self.bos] + ids  # every token is predicted, including the first word

    def logprob(self, texts: list[str]) -> list[float]:
        import torch

        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        device = next(self.model.parameters()).device
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            seqs = [self._ids(t) for t in chunk]
            width = max(map(len, seqs))
            ids = torch.tensor([s + [self.pad] * (width - len(s)) for s in seqs], device=device)
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            token_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
            for text, value in zip(chunk, token_logp.sum(-1).tolist()):
                self.cache[text] = value
        return [self.cache[t] for t in texts]

    def conditional_logprob(self, prompt: str, texts: list[str]) -> list[float]:
        """log P(text | prompt) = log P(prompt + text) - log P(prompt)."""
        base = self.logprob([prompt])[0]
        return [v - base for v in self.logprob([f"{prompt} {t}".rstrip() for t in texts])]

    def clear(self):
        self.cache.clear()
