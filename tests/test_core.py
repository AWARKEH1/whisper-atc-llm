"""Tests that need neither a GPU nor model downloads."""

from atc_asr.biasing_list import align, extract_biasing_words, utterance_blist
from atc_asr.data import cut, to_seconds
from atc_asr.decoding import Candidate, fuse, rank_nbest, select, text_tokens
from atc_asr.normalizer import (aerospace_transform, remove_brackets, remove_spoken_separators,
                                split_numbers_into_digits)
from atc_asr.rerank import choose, rerank
from atc_asr.whisper_utils import hf_to_openai_key

EOT = 50257


# --- normaliser -------------------------------------------------------------
def test_brackets_removed_pair_by_pair():
    # text between two annotations must be kept
    assert remove_brackets("[noise] lufthansa one two [hes] climb").split() == \
        ["lufthansa", "one", "two", "climb"]


def test_aerospace_mapping_and_separators():
    assert aerospace_transform("DLH 4 FL alfa") == "lufthansa four flight level alpha"
    assert remove_spoken_separators("one two decimal five") == "one two five"
    assert split_numbers_into_digits("fl 340") == "fl 3 4 0"


# --- segmentation -----------------------------------------------------------
def test_times_in_samples_or_seconds():
    assert to_seconds(1.5, 3.0) == (1.5, 3.0)
    assert to_seconds(16000, 48000) == (1.0, 3.0)
    assert len(cut(list(range(16000 * 4)), 1.0, 3.0)) == 32000
    assert len(cut(list(range(100)), None, None)) == 100


# --- shallow fusion ---------------------------------------------------------
def test_fused_score_matches_equation():
    assert fuse(-2.0, -10.0, 0.3) == 0.7 * -2.0 + 0.3 * -10.0
    assert fuse(-2.0, -10.0, 0.0) == -2.0


def test_select_keeps_acoustic_score_and_dedupes():
    cands = [Candidate((1, 5), -1.0, 0), Candidate((1, 5), -1.0, 1),   # same prefix twice (first step)
             Candidate((1, EOT), -0.5, 0), Candidate((1, 7), -3.0, 0), Candidate((1, 8), -4.0, 1)]
    fused = [-1.0, -1.0, -0.5, -2.0, -9.0]
    kept, finished = select(cands, fused, beam_size=2, eot=EOT)
    assert [c.tokens for c, _ in kept] == [(1, 5), (1, 7)]
    assert kept[1][0].am == -3.0            # acoustic score is not overwritten by the fused one
    assert finished == {(1, EOT): -0.5}


def test_text_tokens_drop_special_tokens():
    sot = [50258, 50259, 50359, 50363]      # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    assert text_tokens(sot + [11, 12, EOT], sample_begin=4, eot=EOT) == [11, 12]


def test_nbest_is_sorted_by_length_normalised_score():
    finished = {(0, 11, EOT): -1.0,          # 1 token,  -1.0 / 1 = -1.0
                (0, 11, 12, 13, EOT): -2.4,  # 3 tokens, -0.8   <- best
                (0, 14, 15, EOT): -3.0}      # 2 tokens, -1.5
    ranked = rank_nbest(finished, sample_begin=1, eot=EOT)
    assert [t for t, _ in ranked] == [[11, 12, 13], [11], [14, 15]]


# --- reranking --------------------------------------------------------------
class FakeLM:
    def logprob(self, texts):
        return [-float(len(t)) for t in texts]

    def conditional_logprob(self, prompt, texts):
        return self.logprob(texts)


def test_rerank_and_length_norm():
    hyps = ["lufthansa one two", "lufthansa one two three four"]
    assert rerank(hyps, FakeLM()) == "lufthansa one two"
    assert choose(hyps, [-10.0, -12.0], length_norm=True) == 1   # -2.4 per word beats -3.33


# --- biasing list -----------------------------------------------------------
def test_alignment_and_biasing_words():
    assert align("A B C".split(), "A X C".split()) == [("correct", "A"), ("sub", "B"), ("correct", "C")]
    pairs = [("bama one five contact milan", "bahama one five contact milan"),
             ("bama two", "bahama two"), ("contact milan", "contact milan")]
    freq = {"BAMA": 2, "ONE": 50, "FIVE": 50, "CONTACT": 80, "MILAN": 40, "TWO": 60}
    assert extract_biasing_words(pairs, freq) == ["BAMA"]
    assert utterance_blist("BAMA ONE FIVE", {"BAMA", "MILAN"}) == ["BAMA"]


# --- Hugging Face -> OpenAI Whisper weights ---------------------------------
def test_weight_names():
    assert hf_to_openai_key("model.decoder.layers.3.encoder_attn.k_proj.weight") == \
        "decoder.blocks.3.cross_attn.key.weight"
    assert hf_to_openai_key("model.encoder.layers.0.fc1.bias") == "encoder.blocks.0.mlp.0.bias"
    assert hf_to_openai_key("model.encoder.embed_positions.weight") == "encoder.positional_embedding"
    assert hf_to_openai_key("proj_out.weight") is None
