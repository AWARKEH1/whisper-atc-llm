# Train the TCPGen biasing module on top of a frozen Whisper model.
# Adapted from WhisperBiasing (G. Sun, X. Zheng, C. Zhang, P. C. Woodland, 2023), MIT licence:
# https://github.com/BriansIDP/WhisperBiasing  -- requires its modified `whisper` package
# (WhisperBiasing). Changes: defaults for ATC (Whisper-small), output folder created,
# optimiser step on every `accumgrad` batches starting from the first one.

import argparse
import os
import time

import torch
import whisper
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from whisper.model import WhisperBiasing

from dataloader import BiasingProcessor, get_dataloader

parser = argparse.ArgumentParser(description="Train TCPGen for Whisper")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--modeltype", type=str, default="small")
parser.add_argument("--train_json", type=str, required=True)
parser.add_argument("--dev_json", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--nepochs", type=int, default=10)
parser.add_argument("--expdir", type=str, default="exp/tcpgen")
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--decay_pct", type=float, default=1)
parser.add_argument("--warmup_pct", type=float, default=0.0)
parser.add_argument("--log_interval", type=int, default=200)
parser.add_argument("--logfile", type=str, default="")
parser.add_argument("--accumgrad", type=int, default=1)
parser.add_argument("--biasing", action="store_true")
parser.add_argument("--biasinglist", type=str, required=True)
parser.add_argument("--maxKBlen", type=int, default=1)
parser.add_argument("--dropentry", type=float, default=0.0)
parser.add_argument("--attndim", type=int, default=256)
parser.add_argument("--loadfrom", type=str, default="")
parser.add_argument("--GNNtype", type=str, default="none")
parser.add_argument("--GNNdim", type=int, default=0)
parser.add_argument("--useGPT", action="store_true")
args = parser.parse_args()

os.makedirs(args.expdir, exist_ok=True)
logfile = args.logfile or os.path.join(args.expdir, "train.log")


def logging(s):
    print(s)
    with open(logfile, "a+") as f_log:
        f_log.write(s + "\n")


def gpt2_states(target, targetmask):
    """GPT-2 hidden states and distribution aligned with Whisper's target positions."""
    GPTtarget_ids = (target * targetmask)[:, sotlen - 1:-1]
    GPTtarget_ids[:, 0] = GPTtokenizer.bos_token_id  # replace Whisper's BOS with GPT-2's
    GPToutputs = GPTmodel(input_ids=GPTtarget_ids, attention_mask=targetmask[:, sotlen - 1:-1])
    last_hidden = GPToutputs.hidden_states[-1]
    distribution = torch.softmax(GPToutputs.logits, dim=-1)
    # pad GPT-2's distribution to Whisper's vocabulary size
    pad_dist = distribution.new_zeros(distribution.size(0), distribution.size(1),
                                      whisperbiasing.nvocab - distribution.size(2))
    distribution = torch.cat([distribution, pad_dist], dim=-1)
    zero = torch.zeros(last_hidden.size(0), 1, last_hidden.size(-1)).to(model.device)
    last_hidden = torch.cat([zero for _ in range(sotlen - 1)] + [last_hidden, zero], dim=1)
    return last_hidden, distribution


##################
# Model
##################
torch.manual_seed(args.seed)
if args.loadfrom != "":
    whisperbiasing = torch.load(args.loadfrom)
    model = whisperbiasing.whisper
else:
    model = whisper.load_model(args.modeltype)
model.train()
if args.useGPT:
    GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(model.device)
    GPThiddim = GPTmodel.config.n_embd
    GPTtokenizer = GPT2Tokenizer.from_pretrained("gpt2")
else:
    GPThiddim = 0

options = whisper.DecodingOptions(language="en", fp16=False, without_timestamps=True)
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")
decodetask = whisper.decoding.DecodingTask(model, options)
sot_sequence = decodetask.sot_sequence
sotlen = len(sot_sequence)
if args.loadfrom == "":
    whisperbiasing = WhisperBiasing(
        model, tokenizer, model.dims.n_text_state, model.dims.n_text_state, args.attndim,
        model.dims.n_vocab, Bdrop=0.1, biasing=args.biasing, GNNtype=args.GNNtype,
        GNNdim=args.GNNdim, useGPT=args.useGPT, GPThiddim=GPThiddim,
    ).to(model.device)
whisperbiasing.train()

##################
# Data
##################
trainloader = get_dataloader(args.train_json, args.batch_size, loadtarget=True, tokenizer=tokenizer,
                             biasing=args.biasing)
devloader = get_dataloader(args.dev_json, args.batch_size, loadtarget=True, tokenizer=tokenizer,
                           biasing=args.biasing)
biasproc = BiasingProcessor(tokenizer, args.biasinglist, ndistractors=args.maxKBlen, drop=args.dropentry)

##################
# Training
##################
optimiser = Adam(whisperbiasing.parameters(), lr=args.lr)
logging("Start of training")
bestacc = 0
totalstep = args.nepochs * len(trainloader)
for epoch in range(args.nepochs):
    start = time.time()
    totalloss = 0
    optimiser.zero_grad()
    for idx, data in enumerate(trainloader):
        uttnames, fbank, tgt, blist = data
        lextree = biasproc.get_lextree(blist)
        fbank = fbank.to(model.device)
        target = [torch.tensor(list(sot_sequence) + y, dtype=torch.long) for y in tgt]
        target = pad_sequence(target, batch_first=True, padding_value=-100).to(model.device)
        targetmask = target != -100
        GPT_last_hidden, GPT_distribution = None, None
        if args.useGPT:
            with torch.no_grad():
                GPT_last_hidden, GPT_distribution = gpt2_states(target, targetmask)

        loss, _ = whisperbiasing(fbank, target, targetmask, lextree, sotlen,
                                 GPThidden=(GPT_last_hidden, GPT_distribution))
        (loss / args.accumgrad).backward()
        totalloss += loss.item() / args.accumgrad

        if (idx + 1) % args.accumgrad == 0 or idx + 1 == len(trainloader):
            currentstep = epoch * len(trainloader) + idx + 1
            if currentstep > int(args.decay_pct * totalstep):
                factor = (totalstep - currentstep) / (totalstep - int(args.decay_pct * totalstep))
                optimiser.param_groups[0]["lr"] = args.lr * max(0, factor)
            elif currentstep < int(args.warmup_pct * totalstep):
                optimiser.param_groups[0]["lr"] = args.lr * currentstep / int(args.warmup_pct * totalstep)
            optimiser.step()
            optimiser.zero_grad()

        if idx != 0 and idx % args.log_interval == 0:
            logging("{} / {} steps finished in {:.0f}s | Loss: {:.4f} | lr: {}".format(
                idx, len(trainloader), time.time() - start, totalloss / args.log_interval,
                optimiser.param_groups[0]["lr"]))
            totalloss = 0

    # Validation: token accuracy on the dev set
    totalvalset, totalvalacc = 0, 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(devloader):
            uttnames, fbank, tgt, blist = data
            lextree = biasproc.get_lextree(blist)
            fbank = fbank.to(model.device)
            target = [torch.tensor(list(sot_sequence) + y, dtype=torch.long) for y in tgt]
            target = pad_sequence(target, batch_first=True, padding_value=-100).to(model.device)
            targetmask = target != -100
            GPT_last_hidden, GPT_distribution = None, None
            if args.useGPT:
                GPT_last_hidden, GPT_distribution = gpt2_states(target, targetmask)
            loss, output = whisperbiasing(fbank, target, targetmask, lextree, sotlen,
                                          GPThidden=(GPT_last_hidden, GPT_distribution))
            target = target[:, sotlen:]
            output = output.view(target.size(0), target.size(1), -1).max(dim=-1)[1]
            totalvalacc += ((output == target) * targetmask[:, sotlen:]).sum()
            totalvalset += targetmask[:, sotlen:].sum()
    model.train()
    totalacc = (totalvalacc / totalvalset).item()
    logging("Epoch {} | dev token accuracy: {:.4f}".format(epoch + 1, totalacc))
    if totalacc > bestacc:
        torch.save(whisperbiasing, os.path.join(args.expdir, "model.acc.best"))
        bestacc = totalacc
        logging("Saving best model at epoch {}".format(epoch + 1))
    torch.save(whisperbiasing, os.path.join(args.expdir, "snapshot.ep.{}".format(epoch + 1)))
