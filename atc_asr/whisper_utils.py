"""Load OpenAI Whisper, optionally with weights fine-tuned with Hugging Face."""

from __future__ import annotations

import re
from typing import Optional

# Hugging Face -> OpenAI Whisper parameter names
_RENAMES = [
    (r"^model\.", ""),
    (r"^encoder\.layers\.", "encoder.blocks."),
    (r"^decoder\.layers\.", "decoder.blocks."),
    (r"\.self_attn\.q_proj\.", ".attn.query."),
    (r"\.self_attn\.k_proj\.", ".attn.key."),
    (r"\.self_attn\.v_proj\.", ".attn.value."),
    (r"\.self_attn\.out_proj\.", ".attn.out."),
    (r"\.self_attn_layer_norm(?=\.|$)", ".attn_ln"),
    (r"\.encoder_attn\.q_proj\.", ".cross_attn.query."),
    (r"\.encoder_attn\.k_proj\.", ".cross_attn.key."),
    (r"\.encoder_attn\.v_proj\.", ".cross_attn.value."),
    (r"\.encoder_attn\.out_proj\.", ".cross_attn.out."),
    (r"\.encoder_attn_layer_norm(?=\.|$)", ".cross_attn_ln"),
    (r"\.fc1\.", ".mlp.0."),
    (r"\.fc2\.", ".mlp.2."),
    (r"\.final_layer_norm(?=\.|$)", ".mlp_ln"),
    (r"^encoder\.embed_positions\.weight$", "encoder.positional_embedding"),
    (r"^decoder\.embed_positions\.weight$", "decoder.positional_embedding"),
    (r"^decoder\.embed_tokens\.", "decoder.token_embedding."),
    (r"^encoder\.layer_norm\.", "encoder.ln_post."),
    (r"^decoder\.layer_norm\.", "decoder.ln."),
]


def hf_to_openai_key(key: str) -> Optional[str]:
    """OpenAI Whisper name of a HF parameter, or None if OpenAI Whisper has no such parameter
    (the output projection is tied to the token embedding in OpenAI Whisper)."""
    if key.startswith(("proj_out.", "model.lm_head.")):
        return None
    for pattern, repl in _RENAMES:
        key = re.sub(pattern, repl, key)
    return key


def load_whisper(name: str = "small", hf_checkpoint: Optional[str] = None, device: Optional[str] = None):
    """`name` is an OpenAI size ("small"); `hf_checkpoint` a HF fine-tuned model of the same size."""
    import torch
    import whisper

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(name, device=device)
    if hf_checkpoint:
        from transformers import WhisperForConditionalGeneration

        hf_state = WhisperForConditionalGeneration.from_pretrained(hf_checkpoint).state_dict()
        converted = {}
        for k, v in hf_state.items():
            new_key = hf_to_openai_key(k)
            if new_key is not None:
                converted[new_key] = v
        missing, unexpected = model.load_state_dict(converted, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"weight conversion failed: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval()
