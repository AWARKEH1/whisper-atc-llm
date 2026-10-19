"""Second-pass reranking of Whisper's N-best list with a prompted LLM (Li et al., 2023).

Each hypothesis w_1..w_N is scored with the LLM, conditioned on a domain prompt
w_p (report, eq. 5.1):
    LM score = sum_i log P(w_i | w_<i, w_p)
and the best-scoring hypothesis is kept. As in Li et al., the acoustic score is
not added.
"""

from __future__ import annotations

ATC_PROMPT = (
    "You are a professional transcription system specialized in air traffic control (ATC) conversations.\n"
    "Choose the most accurate and domain-correct transcription from the candidates.\n"
    "A correct transcription should:\n"
    "- begin with a valid call sign or airline name and number;\n"
    "- include standard ATC commands (climb/descend to FL, headings, frequency transfers, identification);\n"
    "- use correct frequency formats (e.g., 'one three four decimal six');\n"
    "- be short, structured, no extra words or hallucinations;\n"
    "- respect ATC phraseology: no slang, no filler words, avoid ambiguity.\n\n"
    "Candidate:"
)


def choose(hypotheses: list[str], scores: list[float], length_norm: bool = False) -> int:
    """Index of the best hypothesis (optionally score per word, to avoid favouring short ones)."""
    if length_norm:
        scores = [s / max(1, len(h.split())) for h, s in zip(hypotheses, scores)]
    return max(range(len(hypotheses)), key=lambda k: scores[k])


def rerank(hypotheses: list[str], lm, prompt: str | None = ATC_PROMPT, length_norm: bool = False) -> str:
    hypotheses = [h for h in dict.fromkeys(hypotheses)]  # drop duplicates, keep order
    if len(hypotheses) == 1:
        return hypotheses[0]
    scores = lm.conditional_logprob(prompt, hypotheses) if prompt else lm.logprob(hypotheses)
    return hypotheses[choose(hypotheses, scores, length_norm)]
