# TCPGen data loader.
# Adapted from WhisperBiasing (G. Sun, X. Zheng, C. Zhang, P. C. Woodland, 2023), MIT licence:
# https://github.com/BriansIDP/WhisperBiasing
# Changes: dataset renamed for ATC data; a tokenizer is only required when targets are loaded.

import json
import random

import torch
from torch.utils.data import DataLoader, Dataset

random.seed(1)


class ATCDataset(Dataset):
    """Reads the JSON written by atc_asr.prepare_tcpgen (+ biasing_list assign)."""

    def __init__(self, path, loadtarget=True, tokenizer=None, biasing=True):
        with open(path) as f:
            self.data = json.load(f)
        self.data_idx = list(self.data.keys())
        self.loadtarget = loadtarget
        self.tokenizer = tokenizer
        self.biasing = biasing
        if (loadtarget or biasing) and tokenizer is None:
            raise ValueError("a Whisper tokenizer is required to load targets or biasing lists")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uttname = self.data_idx[index]
        data = self.data[uttname]
        fbank = torch.load(data["fbank"])
        target = data["words"].lower()
        if self.loadtarget:
            if self.biasing:
                # randomly capitalise biasing words, as in the original recipe
                targetwords = []
                for word in target.split():
                    if word.upper() in data["blist"] and random.random() > 0.5:
                        targetwords.append(word.upper()[0:1] + word[1:])
                    else:
                        targetwords.append(word)
                target = " ".join(targetwords)
            target = self.tokenizer.encode(" " + target) + [self.tokenizer.tokenizer.eos_token_id]
        tokenized_words = []
        if self.biasing:
            for word in data["blist"]:
                word = word.lower()
                wordcap = word[0:1].upper() + word[1:]
                tokenized_words.append(tuple(self.tokenizer.encode(" " + word)))
                tokenized_words.append(tuple(self.tokenizer.encode(" " + wordcap)))
        return uttname, fbank, target, tokenized_words


def make_lexical_tree(word_dict, subword_dict, word_unk):
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root


def collate_wrapper(batch):
    uttnames = [i[0] for i in batch]
    fbank = torch.stack([i[1] for i in batch])
    tgt = [i[2] for i in batch]
    blist = []
    for i in batch:
        for word in i[3]:
            if word not in blist:
                blist.append(word)
    return uttnames, fbank, tgt, blist


def get_dataloader(path, bs, shuffle=True, loadtarget=True, tokenizer=None, biasing=False):
    dataset = ATCDataset(path, loadtarget=loadtarget, tokenizer=tokenizer, biasing=biasing)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_fn=collate_wrapper, pin_memory=True)


class BiasingProcessor(object):
    def __init__(self, tokenizer, fulllist, ndistractors=500, drop=0.3):
        self.all_rare_words = []
        with open(fulllist) as fin:
            for line in fin:
                word = line.lower().strip()
                if not word:
                    continue
                self.all_rare_words.append(tuple(tokenizer.encode(" " + word)))
                wordcap = word[0:1].upper() + word[1:]
                self.all_rare_words.append(tuple(tokenizer.encode(" " + wordcap)))
        self.ndistractors = ndistractors
        self.drop = drop
        self.chardict = {idx: idx for idx in range(tokenizer.tokenizer.vocab_size)}

    def insert_distractors(self, uttblist):
        if self.drop > 0:
            uttblist = random.sample(uttblist, int(len(uttblist) * (1 - self.drop)))
        uttblist = [tuple(bword) for bword in uttblist]
        if self.ndistractors < len(self.all_rare_words):
            pool = random.sample(self.all_rare_words, self.ndistractors)
        else:
            pool = self.all_rare_words
        for word in pool:
            if word not in uttblist:
                uttblist.append(word)
        return uttblist[:self.ndistractors]

    def construct_tree(self, uttblist):
        worddict = {word: i + 1 for i, word in enumerate(uttblist)}
        return make_lexical_tree(worddict, self.chardict, -1)

    def get_lextree(self, uttblist):
        return self.construct_tree(self.insert_distractors(uttblist))
