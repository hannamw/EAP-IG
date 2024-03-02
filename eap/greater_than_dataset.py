import random
from typing import List, Union
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

def get_year_indices(tokenizer: PreTrainedTokenizer):
    return torch.tensor([tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])


def get_prob_diff(tokenizer: PreTrainedTokenizer, mean=True):
    year_indices = get_year_indices(tokenizer) 
    def prob_diff(logits, years):
        # Prob diff (negative, since it's a loss)
        probs = torch.softmax(logits[:, -1], dim=-1)[:, year_indices]
        diffs = []
        for prob, year in zip(probs, years):
            diffs.append(prob[year + 1 :].sum() - prob[: year + 1].sum())

        diffs = -torch.stack(diffs).to('cuda') 
        return diffs.mean() if mean else diffs
    return prob_diff

def get_valid_years(
    tokenizer: PreTrainedTokenizer,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(year_toks) for year_toks in tokens]
    valid = torch.tensor([(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized])
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid]

def generate_real_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {year} to the year {century}"
    if eos:
        sentence = "<|endoftext|> " + sentence
    return sentence


def real_sentence_prompt(eos: bool = False) -> List[str]:
    sentence = f"The NOUN lasted from the year XX1 YY to the year XX2".split()
    if eos:
        sentence = ["<|endoftext|>"] + sentence
    return sentence


def generate_bad_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {century}01 to the year {century}"
    if eos:
        sentence = "<|endoftext|> " + sentence
    return sentence


def bad_sentence_prompt(eos: bool = False) -> List[str]:
    sentence = f"The NOUN lasted from the year XX1 01 to the year XX2".split()
    if eos:
        sentence = ["<|endoftext|>"] + sentence
    return sentence


def is_valid_year(year: str, tokenizer) -> bool:
    _year = " " + year
    token = tokenizer(_year)["input_ids"]
    detok = tokenizer.convert_ids_to_tokens(token)
    return len(detok) == 2 and len(detok[1]) == 2


class YearDataset:
    years_to_sample_from: torch.Tensor
    N: int
    ordered: bool
    eos: bool

    nouns: List[str]
    years: torch.Tensor
    years_YY: torch.Tensor
    good_sentences: List[str]
    bad_sentences: List[str]
    good_toks: torch.Tensor
    bad_toks: torch.Tensor
    good_prompt: List[str]
    bad_prompt: List[str]
    good_mask: torch.Tensor
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        years_to_sample_from,
        N: int,
        nouns: Union[str, List[str], Path],
        tokenizer: PreTrainedTokenizer,
        balanced: bool = True,
        eos: bool = False,
        device: str = "cpu",
    ):
        self.years_to_sample_from = years_to_sample_from
        self.N = N
        self.eos=eos

        if isinstance(nouns, str):
            noun_list = [nouns]
        elif isinstance(nouns, list):
            noun_list = nouns
        elif isinstance(nouns, Path):
            with open(nouns, "r") as f:
                noun_list = [line.strip() for line in f]
                noun_list = [noun for noun in noun_list if len(tokenizer(noun).input_ids) == 1]
        else:
            raise ValueError(f"Got bad type of nouns: {type(nouns)}; for nouns: {nouns}")

        self.nouns = random.choices(noun_list, k=N)

        if balanced:
            years = []
            current_year = 2
            years_to_sample_from_YY = self.years_to_sample_from % 100
            for i in range(N):
                sample_pool = self.years_to_sample_from[years_to_sample_from_YY == current_year]
                years.append(sample_pool[random.randrange(len(sample_pool))])
                current_year += 1
                if current_year >= 99:
                    current_year -= 97
            self.years = torch.tensor(years)
        else:
            self.years = torch.tensor(self.years_to_sample_from[torch.randint(0, len(self.years_to_sample_from), (N,))])

        self.years_XX = self.years // 100
        self.years_YY = self.years % 100

        self.good_sentences = [
            generate_real_sentence(noun, int(year.item()), eos=eos) for noun, year in zip(self.nouns, self.years)
        ]
        self.bad_sentences = [
            generate_bad_sentence(noun, int(year.item()), eos=eos) for noun, year in zip(self.nouns, self.years)
        ]

        self.good_prompt = real_sentence_prompt(eos=eos)
        self.bad_prompt = bad_sentence_prompt(eos=eos)

        good_tokenized = tokenizer(self.good_sentences, return_tensors="pt")
        self.good_toks, good_attn = good_tokenized["input_ids"], good_tokenized["attention_mask"]
        assert torch.all(good_attn == 1)
        bad_tokenized = tokenizer(self.bad_sentences, return_tensors="pt")
        self.bad_toks, bad_attn = bad_tokenized["input_ids"], bad_tokenized["attention_mask"]
        assert torch.all(bad_attn == 1)

        # there's a better way to do this
        _good_logits_masks = []
        for year in self.years_YY:
            logits_mask = torch.arange(100)
            _good_logits_masks.append(logits_mask > year)
        self.good_mask = torch.stack(_good_logits_masks)

        self.good_toks = self.good_toks.to(device)
        self.bad_toks = self.bad_toks.to(device)
        self.good_mask = self.good_mask.to(device)

    def __len__(self):
        return self.N

    