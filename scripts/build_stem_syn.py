import json
from collections import defaultdict
from pathlib import Path
from typing import Union, List

import fire
import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from transformers import AutoTokenizer

STEMMER = PorterStemmer()
TOKENIZER = AutoTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)


def get_mask(word_ids: List[int], word_mask: List[int]):
    non_hallucinated_mask = []
    for w_id in word_ids:
        if w_id is None:
            non_hallucinated_mask.append(1)
        else:
            non_hallucinated_mask.append(word_mask[w_id])
    return non_hallucinated_mask


def build_dataset(data_dir: Union[str, Path]):
    outputs = defaultdict(list)
    for ins in tqdm(json.load(open(Path(data_dir) / f"noisy.json")), desc=str(data_dir), dynamic_ncols=True):
        words = {w.lower() for x in ins["src"] for w in nltk.word_tokenize(x)}
        stems = {STEMMER.stem(w) for w in words}
        synsets = set()
        for word in words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name().find("_") < 0:
                        synsets.add(lemma.name())

        match = defaultdict(list)
        for word in ins["tgt"]:
            word = word.lower()
            stem = STEMMER.stem(word)
            match["word"].append(int(word in words))
            match["stem"].append(int(word in words or stem in stems))
            match["syn"].append(int(word in words | synsets or stem in stems))

        word_ids = TOKENIZER(ins["tgt"], is_split_into_words=True).word_ids()
        for key, val in match.items():
            ins["non_hallucinated_mask"] = get_mask(word_ids, val)
            outputs[key].append(dict(ins))
            if len(word_ids) != len(ins["non_hallucinated_mask"]):
                print("ERR!")
    for key, val in outputs.items():
        json.dump(val, open(data_dir / f"noisy_{key}.json", "w"))


def run(data_dir: str = "./data"):
    for data in ("yelp", "amazon", "michelin"):
        build_dataset(Path(data_dir) / data)


if __name__ == '__main__':
    fire.Fire(run)
