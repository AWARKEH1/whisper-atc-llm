"""Error-based biasing list for TCPGen (report, section 6.2).

1. transcribe  - decode the TRAIN split with Whisper (no biasing)
2. freq        - count word frequencies in the TRAIN references
3. extract     - keep words Whisper gets wrong more often than average and that are
                 rare (frequency <= --max-freq) or never seen in training
4. assign      - add each utterance's biasing words ("blist") to a TCPGen JSON

    python -m atc_asr.biasing_list transcribe --metadata m.csv --datapath DATA --out train_hyps.json
    python -m atc_asr.biasing_list freq       --metadata m.csv --out word_freq.txt
    python -m atc_asr.biasing_list extract    --hyps train_hyps.json --freq word_freq.txt --out blist.txt
    python -m atc_asr.biasing_list assign     --json train.json --blist blist.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Levenshtein alignment -> [(op, word)], op in correct / sub / del / ins."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            dp[i][j] = min(dp[i - 1][j - 1] + (0 if same else 1), dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("correct", ref[i - 1])); i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1])); i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1])); i -= 1
        else:
            ops.append(("ins", hyp[j - 1])); j -= 1
    return ops[::-1]


def extract_biasing_words(pairs: list[tuple[str, str]], freq: dict[str, int], max_freq: int = 10) -> list[str]:
    """pairs: normalised (ref, hyp) strings. Returns sorted upper-case words."""
    counts, errors = defaultdict(int), defaultdict(int)
    for ref, hyp in pairs:
        ref_w, hyp_w = ref.upper().split(), hyp.upper().split()
        for w in ref_w:
            counts[w] += 1
        for op, w in align(ref_w, hyp_w):
            if op in ("sub", "del"):
                errors[w] += 1
    if not counts:
        return []
    rate = {w: errors[w] / counts[w] for w in counts}
    mean_rate = sum(rate.values()) / len(rate)
    words = {w for w, r in rate.items() if r > mean_rate and freq.get(w, 0) <= max_freq}
    words |= {w for w in errors if w not in freq}  # never seen in training and misrecognised
    return sorted(words)


def utterance_blist(words: str, biasing: set[str]) -> list[str]:
    return sorted({w for w in words.upper().split() if w in biasing})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe")
    t.add_argument("--metadata", required=True); t.add_argument("--datapath", required=True)
    t.add_argument("--out", required=True); t.add_argument("--whisper", default="small")
    t.add_argument("--hf-checkpoint", default=None)
    f = sub.add_parser("freq")
    f.add_argument("--metadata", required=True); f.add_argument("--out", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--hyps", required=True); e.add_argument("--freq", required=True)
    e.add_argument("--out", required=True); e.add_argument("--max-freq", type=int, default=10)
    a = sub.add_parser("assign")
    a.add_argument("--json", required=True); a.add_argument("--blist", required=True)
    args = p.parse_args()

    from .normalizer import filter_and_normalize

    if args.cmd == "transcribe":
        from tqdm import tqdm
        from whisper.audio import log_mel_spectrogram, pad_or_trim

        from .data import iter_audio, load_metadata
        from .decoding import decode_nbest
        from .whisper_utils import load_whisper

        utts = load_metadata(args.metadata, args.datapath, "train")
        model = load_whisper(args.whisper, args.hf_checkpoint)
        out = {}
        for utt, audio in tqdm(iter_audio(utts), total=len(utts)):
            mel = log_mel_spectrogram(pad_or_trim(audio)).to(model.device)
            hyp = decode_nbest(model, mel, beam_size=1)[0][0]
            out[utt.utt_id] = {"ref": filter_and_normalize(utt.text), "hyp": filter_and_normalize(hyp)}
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"{len(out)} training utterances transcribed -> {args.out}")

    elif args.cmd == "freq":
        import pandas as pd

        df = pd.read_csv(args.metadata)
        df = df[df["split"].astype(str).str.lower() == "train"]
        counter = Counter(w for text in df["text"].dropna()
                          for w in filter_and_normalize(str(text)).upper().split())
        with open(args.out, "w", encoding="utf-8") as fout:
            for w, c in counter.most_common():
                fout.write(f"{w} {c}\n")
        print(f"{len(counter)} words -> {args.out}")

    elif args.cmd == "extract":
        data = json.load(open(args.hyps))
        freq = {}
        for line in open(args.freq, encoding="utf-8"):
            w, c = line.split()
            freq[w.upper()] = int(c)
        words = extract_biasing_words([(d["ref"], d["hyp"]) for d in data.values()], freq, args.max_freq)
        open(args.out, "w").write("".join(w + "\n" for w in words))
        print(f"{len(words)} biasing words -> {args.out}")

    elif args.cmd == "assign":
        biasing = {filter_and_normalize(w).upper() for w in open(args.blist) if w.strip()}
        data = json.load(open(args.json))
        for entry in data.values():
            entry["blist"] = utterance_blist(entry.get("words", ""), biasing)
        json.dump(data, open(args.json, "w"), indent=2)
        print(f"blist added to {len(data)} utterances in {args.json}")


if __name__ == "__main__":
    main()
