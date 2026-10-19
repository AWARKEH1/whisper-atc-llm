"""ATC text normalisation applied to references AND hypotheses before WER.

Steps (same order as in the internship report, section 3.2):
 1. remove bracketed annotations [..] and <..>
 2. remove non-alphanumeric characters
 3. separate digits from letters
 4. aerospace terminology mapping (NATO alphabet, call-sign codes, variants)
 5. remove spoken separators ("decimal", "comma", "point")
 6. Whisper English normaliser, applied twice (number words -> digits, then
    leading zeros removed)
 7. split numbers into single digits
 8. split greetings ("goodbye" -> "good bye")
 9. lowercase + standard spelling variants ("centre" -> "center", ...)
10. final normaliser + digit split
"""

from __future__ import annotations

import re
from functools import lru_cache

NATO = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo", "F": "foxtrot",
    "G": "golf", "H": "hotel", "I": "india", "J": "juliett", "K": "kilo", "L": "lima",
    "M": "mike", "N": "november", "O": "oscar", "P": "papa", "Q": "quebec", "R": "romeo",
    "S": "sierra", "T": "tango", "U": "uniform", "V": "victor", "W": "whiskey", "X": "xray",
    "Y": "yankee", "Z": "zulu",
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
    "8": "eight", "9": "nine", "10": "ten", "0": "zero", "00": "hundred", "000": "thousand",
    ".": "decimal", ",": "comma", "-": "dash",
}
NATO_VARIANTS = {"alfa": "alpha", "oskar": "oscar", "ekko": "echo", "gulf": "golf"}
TERMINOLOGY = {"FL": "flight level"}
TEXT_VARIANTS = {"descent": "descend"}
# Whisper sometimes writes "eurowings seven alpha bravo" as "EW7AB"
AIRLINES_IATA = {"BA": "british airways", "KL": "klm", "LH": "lufthansa", "EW": "eurowings"}
AIRLINES_ICAO = {"BAW": "british airways", "DLH": "lufthansa", "KLM": "klm", "EWG": "eurowings"}
SPOKEN_SEPARATORS = {"decimal", "comma", "point"}
SPELLING = [("lineup", "line up"), ("centre", "center"), ("k l m", "klm"),
            ("niner", "nine"), ("x-ray", "xray")]


@lru_cache(maxsize=1)
def _whisper_normalizer():
    from whisper.normalizers import EnglishTextNormalizer
    return EnglishTextNormalizer()


def remove_brackets(text: str) -> str:
    """Remove every [..] and <..> annotation (each pair separately)."""
    return re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)


def remove_non_alphanumeric(text: str) -> str:
    return "".join(c for c in text if c.isalnum() or c == " ")


def separate_numbers_and_text(text: str) -> str:
    return " ".join(re.split(r"(\d+)", text))


def _map_word(word: str) -> str:
    out = word
    for table, key in ((NATO, word), (NATO_VARIANTS, word.lower()), (TERMINOLOGY, word),
                       (TEXT_VARIANTS, word.lower()), (AIRLINES_IATA, word.upper()),
                       (AIRLINES_ICAO, word.upper())):
        if key in table:
            out = table[key]
    return out


def aerospace_transform(text: str) -> str:
    return " ".join(_map_word(w) for w in text.split())


def remove_spoken_separators(text: str) -> str:
    return " ".join(w for w in text.split() if w.lower() not in SPOKEN_SEPARATORS)


def split_numbers_into_digits(text: str) -> str:
    return " ".join(" ".join(w) if w.isdigit() and w.isascii() else w for w in text.split())


def split_greetings(text: str) -> str:
    return " ".join("good bye" if w.lower() == "goodbye" else w for w in text.split())


def standard_words(text: str) -> str:
    text = text.lower()
    for old, new in SPELLING:
        text = text.replace(old, new)
    return text


def filter_and_normalize(text: str) -> str:
    norm = _whisper_normalizer()
    text = remove_brackets(str(text))
    text = remove_non_alphanumeric(text)
    text = separate_numbers_and_text(text)
    text = aerospace_transform(text)
    text = remove_spoken_separators(text)
    text = norm(norm(text))
    text = split_numbers_into_digits(text)
    text = split_greetings(text)
    text = standard_words(text)
    text = norm(text)
    text = split_numbers_into_digits(text)
    return " ".join(text.split())
