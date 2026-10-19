"""Build the JSON index + log-Mel features that TCPGen (tcpgen/train.py, decode.py) reads.

Each speech turn is cut from its recording first (ATCO2 files hold several
turns) and gets its own id, then:
    {utt_id: {"fbank": path.pt, "words": NORMALISED TEXT, "blist": []}}
Fill "blist" afterwards with `python -m atc_asr.biasing_list assign`.

    python -m atc_asr.prepare_tcpgen --metadata m.csv --datapath DATA --split train \
        --fbank-dir fbank/ --out train.json
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--datapath", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--fbank-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from tqdm import tqdm
    from whisper.audio import log_mel_spectrogram, pad_or_trim

    from .data import iter_audio, load_metadata
    from .normalizer import filter_and_normalize

    os.makedirs(args.fbank_dir, exist_ok=True)
    utts = load_metadata(args.metadata, args.datapath, args.split)
    index = {}
    for utt, audio in tqdm(iter_audio(utts), total=len(utts)):
        path = os.path.join(args.fbank_dir, f"{utt.utt_id}.pt")
        torch.save(log_mel_spectrogram(pad_or_trim(audio)), path)
        index[utt.utt_id] = {"fbank": path, "words": filter_and_normalize(utt.text).upper(), "blist": []}
    json.dump(index, open(args.out, "w"), indent=2)
    print(f"{len(index)} utterances ({args.split}) -> {args.out}")


if __name__ == "__main__":
    main()
