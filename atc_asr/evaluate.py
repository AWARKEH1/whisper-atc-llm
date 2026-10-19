"""Evaluate Whisper on ATC speech: plain beam search, shallow fusion or LLM reranking.

Examples:
    python -m atc_asr.evaluate --metadata atco2.csv --datapath /data/ATCO2 --split test \
        --method baseline --beam-size 5
    python -m atc_asr.evaluate ... --method sf --lm mistralai/Mistral-7B-Instruct-v0.2 --lm-weight 0.3
    python -m atc_asr.evaluate ... --method rerank --lm meta-llama/Llama-3.2-3B-Instruct --beam-size 10

Choose --lm-weight / --beam-size on --split validation, then report --split test.
"""

from __future__ import annotations

import argparse
import json
import os


def wer_report(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    from .normalizer import filter_and_normalize

    refs_n = [filter_and_normalize(r) for r in refs]
    hyps_n = [filter_and_normalize(h) for h in hyps]
    keep = [i for i, r in enumerate(refs_n) if r]  # a reference empty after normalisation has no words
    refs_n, hyps_n = [refs_n[i] for i in keep], [hyps_n[i] for i in keep]
    return {
        "n_utterances": len(keep),
        "wer": round(100 * jiwer.wer(refs_n, hyps_n), 2),
        "cer": round(100 * jiwer.cer(refs_n, hyps_n), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="CSV: file_name,text,split[,audio_start,audio_end]")
    p.add_argument("--datapath", required=True, help="root folder of the audio files")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--method", required=True, choices=["baseline", "sf", "rerank"])
    p.add_argument("--whisper", default="small", help="OpenAI Whisper size")
    p.add_argument("--hf-checkpoint", default=None, help="fine-tuned HF Whisper (same size)")
    p.add_argument("--beam-size", type=int, default=5, help="beam size = N-best size for rerank")
    p.add_argument("--lm", default=None, help="HF causal LM for sf / rerank")
    p.add_argument("--lm-adapter", default=None, help="LoRA adapter for the LM (e.g. TinyLlama fine-tuned)")
    p.add_argument("--lm-weight", type=float, default=0.3, help="lambda for shallow fusion")
    p.add_argument("--no-prompt", action="store_true", help="rerank without the ATC prompt")
    p.add_argument("--length-norm", action="store_true", help="rerank with score per word")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    if args.method != "baseline" and not args.lm:
        p.error("--lm is required for sf and rerank")

    from tqdm import tqdm
    from whisper.audio import log_mel_spectrogram, pad_or_trim

    from .data import iter_audio, load_metadata
    from .decoding import decode_nbest
    from .lm import load_lm
    from .rerank import ATC_PROMPT, rerank
    from .whisper_utils import load_whisper

    utts = load_metadata(args.metadata, args.datapath, args.split, args.limit)
    model = load_whisper(args.whisper, args.hf_checkpoint)
    lm = load_lm(args.lm, args.lm_adapter) if args.lm else None
    print(f"{len(utts)} utterances | method={args.method} beam={args.beam_size}"
          + (f" lambda={args.lm_weight}" if args.method == "sf" else ""))

    rows = []
    for utt, audio in tqdm(iter_audio(utts), total=len(utts)):
        mel = log_mel_spectrogram(pad_or_trim(audio)).to(model.device)
        nbest = decode_nbest(model, mel, args.beam_size,
                             lm=lm if args.method == "sf" else None,
                             lam=args.lm_weight if args.method == "sf" else 0.0)
        hyps = [text for text, _ in nbest]
        if args.method == "rerank":
            hyp = rerank(hyps, lm, None if args.no_prompt else ATC_PROMPT, args.length_norm)
            lm.clear()
        else:
            hyp = hyps[0]
        rows.append({"id": utt.utt_id, "ref": utt.text, "hyp": hyp, "nbest": hyps})

    metrics = {**vars(args), **wer_report([r["ref"] for r in rows], [r["hyp"] for r in rows])}
    name = f"{args.split}_{args.method}_b{args.beam_size}" + (f"_l{args.lm_weight}" if args.method == "sf" else "")
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, name + ".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, name + ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"WER {metrics['wer']} %  |  CER {metrics['cer']} %  ({metrics['n_utterances']} utterances)")


if __name__ == "__main__":
    main()
