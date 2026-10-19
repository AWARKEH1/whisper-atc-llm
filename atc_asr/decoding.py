"""Whisper beam search with optional shallow fusion, returning the full N-best list.

Shallow fusion (report, eq. 4.1): every candidate y is ranked with
    score(y) = (1 - lambda) * log P_AM(y | x) + lambda * log P_LM(y)
where log P_AM is Whisper's accumulated log-probability and log P_LM the LLM's
log-probability of the candidate's text. The two scores are kept separately:
only the acoustic score is accumulated step by step, the fused score is used
to choose which candidates stay in the beam and to rank finished hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class Candidate:
    tokens: tuple
    am: float        # accumulated Whisper log-probability
    source: int      # beam row it extends


def fuse(am: float, lm: float, lam: float) -> float:
    return (1.0 - lam) * am + lam * lm


def select(cands: list[Candidate], fused: list[float], beam_size: int, eot: int):
    """Best `beam_size` unfinished candidates + all finished ones among the better ones.

    Same rule as Whisper's beam search, applied to the fused score: candidates
    are visited from best to worst, ending ones are stored as finished, the
    others fill the beam until it holds `beam_size` hypotheses.
    """
    order = sorted(range(len(cands)), key=lambda k: fused[k], reverse=True)
    kept, finished, seen = [], {}, set()
    for k in order:
        c = cands[k]
        if c.tokens in seen:
            continue
        seen.add(c.tokens)
        if c.tokens[-1] == eot:
            finished[c.tokens] = fused[k]
        else:
            kept.append((c, fused[k]))
            if len(kept) == beam_size:
                break
    return kept, finished


def text_tokens(seq, sample_begin: int, eot: int) -> list[int]:
    """Tokens of the transcript only: no <|startoftranscript|>, language, task or EOT."""
    out = []
    for t in list(seq)[sample_begin:]:
        if t >= eot:
            break
        out.append(t)
    return out


def rank_nbest(finished: dict, sample_begin: int, eot: int) -> list[tuple[list[int], float]]:
    """Finished hypotheses sorted by length-normalised score (Whisper's default ranking)."""
    items = []
    for seq, score in finished.items():
        toks = text_tokens(seq, sample_begin, eot)
        items.append((toks, score / max(1, len(toks))))  # Whisper's MaximumLikelihoodRanker
    return sorted(items, key=lambda x: x[1], reverse=True)


def make_decoder_class():
    """Built lazily so that the pure functions above can be tested without Whisper."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from whisper.decoding import BeamSearchDecoder

    class FusionBeamSearchDecoder(BeamSearchDecoder):
        def __init__(self, beam_size, eot, inference, patience=None, lm_score: Optional[Callable] = None,
                     lam: float = 0.0, decode_text: Optional[Callable] = None, sample_begin: int = 0):
            super().__init__(beam_size, eot, inference, patience)
            self.lm_score, self.lam = lm_score, lam
            self.decode_text, self.sample_begin = decode_text, sample_begin
            self.row_fused = None

        def reset(self):
            super().reset()
            self.row_fused = None

        def _fused(self, cands: list[Candidate]) -> list[float]:
            if self.lm_score is None or self.lam == 0:
                return [c.am for c in cands]
            texts = [self.decode_text(text_tokens(c.tokens, self.sample_begin, self.eot)) for c in cands]
            return [fuse(c.am, lm, self.lam) for c, lm in zip(cands, self.lm_score(texts))]

        def update(self, tokens, logits, sum_logprobs):
            if tokens.shape[0] % self.beam_size != 0:
                raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")
            n_audio = tokens.shape[0] // self.beam_size
            if self.finished_sequences is None:
                self.finished_sequences = [{} for _ in range(n_audio)]
            logprobs = F.log_softmax(logits.float(), dim=-1)

            next_tokens, sources, new_am, new_fused, step_finished = [], [], [], [], []
            for i in range(n_audio):
                cands = []
                for j in range(self.beam_size):
                    idx = i * self.beam_size + j
                    prefix = tokens[idx].tolist()
                    top_lp, top_tok = logprobs[idx].topk(self.beam_size + 1)
                    for lp, tok in zip(top_lp.tolist(), top_tok.tolist()):
                        cands.append(Candidate(tuple(prefix + [tok]), sum_logprobs[idx].item() + lp, idx))
                kept, finished = select(cands, self._fused(cands), self.beam_size, self.eot)
                for c, f in kept:
                    next_tokens.append(list(c.tokens))
                    sources.append(c.source)
                    new_am.append(c.am)
                    new_fused.append(f)
                step_finished.append(finished)

            sum_logprobs.copy_(torch.tensor(new_am, device=sum_logprobs.device))  # acoustic score only
            self.row_fused = torch.tensor(new_fused)
            tokens = torch.tensor(next_tokens, device=tokens.device)
            self.inference.rearrange_kv_cache(sources)

            for prev, new in zip(self.finished_sequences, step_finished):
                for seq in sorted(new, key=new.get, reverse=True):
                    if len(prev) >= self.max_candidates:
                        break
                    prev[seq] = new[seq]
            completed = all(len(s) >= self.max_candidates for s in self.finished_sequences)
            return tokens, completed

        def finalize(self, preceding_tokens, sum_logprobs):
            scores = (self.row_fused.reshape(sum_logprobs.shape) if self.row_fused is not None
                      else sum_logprobs).cpu()
            for i, sequences in enumerate(self.finished_sequences):
                if len(sequences) < self.beam_size:  # not enough finished: add the best unfinished ones
                    for j in list(np.argsort(scores[i]))[::-1]:
                        sequences[tuple(preceding_tokens[i, j].tolist() + [self.eot])] = scores[i][j].item()
                        if len(sequences) >= self.beam_size:
                            break
            tokens = [[torch.tensor(seq) for seq in s.keys()] for s in self.finished_sequences]
            return tokens, [list(s.values()) for s in self.finished_sequences]

    return FusionBeamSearchDecoder


def decode_nbest(model, mel, beam_size: int = 5, lm=None, lam: float = 0.0, language: str = "en"):
    """N-best list [(text, score)] for one 30-s mel spectrogram, best first.

    lm: an LMScorer for shallow fusion (None = plain Whisper beam search).
    """
    from whisper.decoding import DecodingOptions, DecodingTask

    options = DecodingOptions(language=language, beam_size=beam_size, fp16=False, without_timestamps=True)
    task = DecodingTask(model, options)
    tokenizer = task.tokenizer
    Decoder = make_decoder_class()
    task.decoder = Decoder(beam_size, tokenizer.eot, task.inference, options.patience,
                           lm_score=lm.logprob if lm is not None else None, lam=lam,
                           decode_text=lambda toks: tokenizer.decode(toks).strip(),
                           sample_begin=task.sample_begin)
    if lm is not None:
        lm.clear()
    task.run(mel.unsqueeze(0) if mel.ndim == 2 else mel)
    ranked = rank_nbest(task.decoder.finished_sequences[0], task.sample_begin, tokenizer.eot)
    return [(tokenizer.decode(toks).strip(), score) for toks, score in ranked]
