"""Metadata loading and audio segmentation for ATCOSIM / ATCO2.

The metadata CSV has one row per speech turn:
    file_name, text, split[, role, airport, audio_start, audio_end]
ATCOSIM: one turn per file (no start/end). ATCO2: several turns per file, cut
with audio_start / audio_end (seconds, or samples for some exports).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

SAMPLE_RATE = 16000


@dataclass
class Utterance:
    utt_id: str
    audio_path: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    role: str = ""
    airport: str = ""


def _to_float(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else value  # NaN -> None


def to_seconds(start: Optional[float], end: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Times above 1000 are sample indices (16 kHz), otherwise seconds."""
    if (start or 0) > 1000 or (end or 0) > 1000:
        return (None if start is None else start / SAMPLE_RATE,
                None if end is None else end / SAMPLE_RATE)
    return start, end


def load_metadata(metadata: str, datapath: str, split: str, limit: int = 0) -> list[Utterance]:
    """Rows of one split with a non-empty transcript and an existing audio file.

    Each turn gets a unique id (file stem + turn index), so several turns from
    the same ATCO2 recording never overwrite each other.
    """
    import pandas as pd

    df = pd.read_csv(metadata)
    df = df[df["split"].astype(str).str.lower() == split.lower()]
    df = df[df["text"].notna() & df["text"].astype(str).str.strip().ne("")]
    utts, turn_count = [], {}
    for _, row in df.iterrows():
        path = os.path.join(datapath, row["file_name"])
        if not os.path.exists(path):
            continue
        stem = os.path.splitext(os.path.basename(row["file_name"]))[0]
        turn = turn_count.get(stem, 0)
        turn_count[stem] = turn + 1
        start, end = to_seconds(_to_float(row.get("audio_start")), _to_float(row.get("audio_end")))
        utts.append(Utterance(f"{stem}_{turn:03d}", path, str(row["text"]), start, end,
                              str(row.get("role", "") or ""), str(row.get("airport", "") or "")))
        if limit and len(utts) >= limit:
            break
    return utts


def cut(audio, start: Optional[float], end: Optional[float]):
    """Slice a 16 kHz waveform to [start, end] seconds (whole file if no times)."""
    n = len(audio)
    a = 0 if start is None else max(0, min(n, int(start * SAMPLE_RATE)))
    b = n if end is None else max(a, min(n, int(end * SAMPLE_RATE)))
    return audio[a:b]


def iter_audio(utts: list[Utterance]) -> Iterator[tuple[Utterance, "object"]]:
    """Yield (utterance, float32 waveform of its segment); each file is read once in a row."""
    from whisper.audio import load_audio

    cached_path, cached_audio = None, None
    for utt in utts:
        if utt.audio_path != cached_path:
            cached_path, cached_audio = utt.audio_path, load_audio(utt.audio_path)
        yield utt, cut(cached_audio, utt.start, utt.end)
