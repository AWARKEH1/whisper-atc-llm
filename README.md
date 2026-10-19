# Whisper + LLMs for Air Traffic Control speech recognition

> Can a large language model help Whisper transcribe noisy, accented, jargon-heavy air-traffic-control radio?
> Three ways of adding language knowledge to **Whisper-small** — **shallow fusion**, **LLM reranking** and **TCPGen contextual biasing** — tested alone and combined on the **ATCOSIM** and **ATCO2** corpora.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Whisper](https://img.shields.io/badge/OpenAI-Whisper-black) ![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow) ![HPC](https://img.shields.io/badge/HPC-Grid'5000-green)

Master's thesis internship — **Inria Nancy / LORIA, Multispeech team** (March–September 2025), MSc NLP, Université de Lorraine.

## The problem

ATC communications are hard for general-purpose ASR: VHF radio noise, fast speech, non-native English and a very specific vocabulary (call signs, flight levels, frequencies, waypoints). Whisper, trained on web audio, makes errors that change the meaning:

```
reference : merair six nine four four climb to flight level three three zero
whisper   : mirror six nine four four climb to flight level three three zero
```

## Methods

| Method | When it acts | Idea |
|---|---|---|
| **Shallow fusion** | during beam search | each candidate is ranked with `(1 − λ)·log P_whisper + λ·log P_LLM` |
| **Second-pass reranking** | after decoding | an LLM, given an ATC prompt, picks the best of Whisper's N-best hypotheses (Li et al., 2023) |
| **TCPGen** | during decoding | a pointer-generator copies words from a biasing list stored as a prefix tree (Sun et al., 2023) |

The biasing list is **error-based**: Whisper decodes the training set, and the words it gets wrong more often than average (and that are rare or unseen) are kept — 588 words, e.g. `RUNWAY, ECHO, CLEARED, QUEBEC, TAKEOFF…`.

LLMs compared: **TinyLlama-1.1B** (also LoRA fine-tuned on ATC text), **Llama-3.2-3B**, **Mistral-7B**, **GPT-2**.

## Results

*From the Master's thesis (experiments run on Grid'5000 during the internship, full ATCOSIM and ATCO2 corpora, Whisper-small, WER % after ATC normalisation).*

### ATCO2 — real radio recordings

| System | Whisper-small | Whisper-small fine-tuned on ATCO2 |
|---|---|---|
| Whisper alone | 71.3 | 28.57 |
| + shallow fusion (Mistral-7B) | 65.64 (λ = 0.3) | 19.87 (λ = 0.7) |
| + TCPGen | 59.4 | 56.4 (TCPGen trained on ATCOSIM) |
| + TCPGen + GPT-2 + reranking | **44.2** | 46.5 |
| + GPT-2 + reranking | 63.6 | **18.6** |

### ATCOSIM — simulated, clean speech (Whisper-small, not fine-tuned)

| System | WER |
|---|---|
| Whisper alone | 30.66 |
| + shallow fusion (Mistral-7B, λ = 0.3) | 26.60 |
| + reranking, no prompt (Mistral-7B, N = 10) | 27.24 |
| + reranking with ATC prompt (Llama-3.2-3B, N = 10) | **24.85** |

### What we learned

- **Fine-tuning Whisper is the biggest single gain** (ATCO2: 71.3 → 28.57); language models add on top of it.
- **Shallow fusion and reranking are safe additions**: they never change Whisper's weights and help both with and without fine-tuning. The best fine-tuned system, Whisper + GPT-2 + reranking, reaches **18.6 % WER**.
- **A domain prompt helps reranking**, more so with a longer N-best list (N ≥ 8), because the right transcription is more often in the list.
- **TCPGen helps a non-adapted Whisper a lot** (71 → 59.4, 44.2 combined) **but hurts a fine-tuned one**: trained on another corpus (ATCOSIM), it pushes biasing words where there is no acoustic evidence and produces insertions and repetitions.

## Repository

```
atc_asr/
├── normalizer.py     # ATC text normalisation for WER (NATO alphabet, call signs, digits…)
├── data.py           # metadata CSV, per-turn audio segments (ATCO2 files hold several turns)
├── whisper_utils.py  # load Whisper, inject weights fine-tuned with Hugging Face
├── decoding.py       # beam search with shallow fusion, full N-best list
├── lm.py             # exact sentence log-probabilities from a causal LLM
├── rerank.py         # second-pass reranking with the ATC prompt
├── evaluate.py       # CLI: baseline / sf / rerank -> WER, CER, hypotheses
├── biasing_list.py   # error-based biasing list (transcribe, freq, extract, assign)
├── prepare_tcpgen.py # log-Mel features + JSON index for TCPGen
└── finetune_lm.py    # LoRA fine-tuning of TinyLlama on ATC transcripts
tcpgen/               # TCPGen training / decoding, adapted from WhisperBiasing
tests/                # unit tests (no GPU needed)
```

## Usage

```bash
pip install -r requirements.txt

# Whisper alone, then shallow fusion and reranking (choose λ and N on --split validation)
python -m atc_asr.evaluate --metadata atco2.csv --datapath /data/ATCO2 --split test --method baseline
python -m atc_asr.evaluate --metadata atco2.csv --datapath /data/ATCO2 --split test --method sf \
       --lm mistralai/Mistral-7B-Instruct-v0.2 --lm-weight 0.7 --hf-checkpoint models/whisper-small-atco2
python -m atc_asr.evaluate --metadata atcosim.csv --datapath /data/ATCOSIM --split test --method rerank \
       --lm meta-llama/Llama-3.2-3B-Instruct --beam-size 10

# TCPGen: biasing list -> features -> training -> decoding
python -m atc_asr.biasing_list transcribe --metadata atco2.csv --datapath /data/ATCO2 --out train_hyps.json
python -m atc_asr.biasing_list freq    --metadata atco2.csv --out word_freq.txt
python -m atc_asr.biasing_list extract --hyps train_hyps.json --freq word_freq.txt --out blist.txt
python -m atc_asr.prepare_tcpgen --metadata atco2.csv --datapath /data/ATCO2 --split train --fbank-dir fbank --out train.json
python -m atc_asr.biasing_list assign  --json train.json --blist blist.txt
python tcpgen/train.py  --train_json train.json --dev_json dev.json --biasinglist blist.txt --biasing
python tcpgen/decode.py --test_json test.json --biasinglist blist.txt --biasing --loadfrom exp/tcpgen/model.acc.best

pytest
```

The metadata CSV has one row per speech turn: `file_name, text, split` and, for ATCO2, `audio_start, audio_end`.
`tcpgen/` needs the modified `whisper` package of [WhisperBiasing](https://github.com/BriansIDP/WhisperBiasing). Gated models (Llama, Mistral) read the Hugging Face token from the `HF_TOKEN` environment variable.

## Data

- **ATCOSIM** (Hofbauer et al., 2008): about 9,500 utterances of simulated en-route control, clean microphone speech, free ([also on Hugging Face](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus)).
- **ATCO2** (Zuluaga-Gomez et al., 2022): real VHF recordings from European airports; the full corpus is commercial, a 1-hour sample is free ([Hugging Face](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h)).

No audio is included in this repository.

## Credits

- Internship supervised by **Irina Illina** and **Raphaël Bagat** (Multispeech, Inria / LORIA). Experiments run on the **Grid'5000** testbed.
- TCPGen code adapted from [WhisperBiasing](https://github.com/BriansIDP/WhisperBiasing) (MIT) — G. Sun, X. Zheng, C. Zhang, P. C. Woodland, *Can Contextual Biasing Remain Effective with Whisper and GPT-2?*, Interspeech 2023.
- Reranking: Li et al. (2023), *Prompting Large Language Models for Zero-Shot Domain Adaptation in Speech Recognition*. Shallow fusion: Kannan et al. (2018).
- Whisper: Radford et al. (2023), MIT licence.
