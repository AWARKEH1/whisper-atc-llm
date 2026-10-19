# Decode with Whisper + TCPGen (and optionally GPT-2 shallow fusion).
# Adapted from WhisperBiasing (G. Sun, X. Zheng, C. Zhang, P. C. Woodland, 2023), MIT licence:
# https://github.com/BriansIDP/WhisperBiasing  -- requires its modified `whisper` package.
# Changes: WER computed with the same ATC normaliser as atc_asr.evaluate, ILME model of the
# same size as the decoding model, Whisper-small default.

import argparse
import json
import os
import sys
import time

import editdistance
import torch
import whisper
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataloader import BiasingProcessor, get_dataloader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from atc_asr.normalizer import filter_and_normalize  # noqa: E402

parser = argparse.ArgumentParser(description="Whisper + TCPGen decoding")
parser.add_argument("--test_json", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--beamsize", type=int, default=3)
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--expdir", type=str, default="exp/tcpgen")
parser.add_argument("--loadfrom", type=str, default="", help="trained TCPGen model (model.acc.best)")
parser.add_argument("--biasing", action="store_true")
parser.add_argument("--use_gpt2", action="store_true")
parser.add_argument("--save_nbest", action="store_true")
parser.add_argument("--lm_weight", type=float, default=0)
parser.add_argument("--ilm_weight", type=float, default=0)
parser.add_argument("--biasinglist", type=str, required=True)
parser.add_argument("--maxKBlen", type=int, default=1)
parser.add_argument("--dropentry", type=float, default=0.0)
parser.add_argument("--modeltype", type=str, default="small")
args = parser.parse_args()
os.makedirs(args.expdir, exist_ok=True)

GPTmodel, GPTtokenizer = None, None
if args.use_gpt2:
    GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(args.device)

if args.loadfrom:
    biasing_model = torch.load(args.loadfrom)
    biasing_model.eval()
    model = biasing_model.whisper
    useGPT = getattr(biasing_model, "useGPT", False)
else:
    model = whisper.load_model(args.modeltype).eval()
    biasing_model = None
    useGPT = False
if useGPT or args.use_gpt2:
    GPTtokenizer = GPT2Tokenizer.from_pretrained("gpt2")

ilme_model = None
if args.use_gpt2 and args.ilm_weight > 0:
    ilme_model = whisper.load_model(args.modeltype).eval()
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")

testloader = get_dataloader(args.test_json, args.eval_batch_size, loadtarget=False, tokenizer=tokenizer,
                            biasing=args.biasing, shuffle=False)
biasproc = BiasingProcessor(tokenizer, args.biasinglist, ndistractors=args.maxKBlen, drop=args.dropentry)

refs, hyps, nbest_dict = [], [], {}
start = time.time()
for idx, (uttnames, fbank, tgt, blist) in enumerate(testloader):
    fbank = fbank.to(model.device)
    origtree = biasproc.get_lextree(blist)
    if biasing_model is not None and getattr(biasing_model, "GNN", None) is not None:
        biasing_model.GNN(origtree, model.decoder.token_embedding)

    options = whisper.DecodingOptions(
        language="en", without_timestamps=True, beam_size=args.beamsize, biasing=args.biasing,
        biasingmodule=biasing_model, origtree=origtree, fp16=False, shallowfusion=args.use_gpt2,
        useGPT=useGPT, GPT2=GPTmodel, lm_weight=args.lm_weight, GPT2tokenizer=GPTtokenizer,
        ilm_weight=args.ilm_weight, ilme_model=ilme_model,
    )
    result = whisper.decode(model, fbank, options)
    for i, ref in enumerate(tgt):
        refs.append((uttnames[i], ref))
        hyps.append(result[i].text)
        if args.save_nbest:
            nbest_dict[uttnames[i]] = [{"text": t, "whisper_slp": s}
                                       for t, s in zip(result[i].text_nbest, result[i].sum_logprob_nbest)]
    if idx % 50 == 0 and idx > 0:
        print("{} / {} utterances | {:.0f}s".format(idx, len(testloader), time.time() - start))

# WER with the ATC normaliser (same as atc_asr.evaluate)
errors, words = 0, 0
with open(os.path.join(args.expdir, "hyp_ref.jsonl"), "w", encoding="utf-8") as fout:
    for (uttname, ref), hyp in zip(refs, hyps):
        ref_n, hyp_n = filter_and_normalize(ref).split(), filter_and_normalize(hyp).split()
        errors += editdistance.eval(hyp_n, ref_n)
        words += len(ref_n)
        fout.write(json.dumps({"id": uttname, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
print("WER: {}/{} = {:.2f} %".format(errors, words, 100 * errors / max(1, words)))

if args.save_nbest:
    with open(os.path.join(args.expdir, "nbest.json"), "w") as fout:
        json.dump(nbest_dict, fout)
