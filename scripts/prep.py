import gzip
import json
import os
import re
import shutil
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Union

import fasttext
import fire
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from unidecode import unidecode

FEWSUM_DIR = "FewSum/artifacts/{}/gold_summs/"


def strip_text(s: str) -> str:
    s = unidecode(s)
    s = re.sub(r"!{2,}", "! ", s)
    s = re.sub(r"\?{2,}", "? ", s)
    s = re.sub(r",{2,}", ", ", s)
    s = re.sub(r"\.{2,}", ". ", s)
    s = re.sub(r"-{2,}", "-", s)
    return " ".join(s.split())


def build_gold(df: pd.DataFrame, output_file: Union[str, Path]):
    outputs = []
    for _, ins in df.iterrows():
        out = {
            "cat": ins["cat"],
            "business_id": ins["group_id"],
            "src": [strip_text(ins[f"rev{i + 1}"]) for i in range(8)],
            "ref": [ins[f"summ{i + 1}"] for i in range(3)]
        }
        for tgt in out["ref"]:
            out["tgt"] = nltk.word_tokenize(strip_text(tgt))
            outputs.append(dict(out))
    json.dump(outputs, open(output_file, "w"))
    return df.group_id


def process_gold(gold_dir: Union[str, Path], output_dir: Union[str, Path]):
    gold_dir, output_dir = Path(gold_dir), Path(output_dir)
    gold_summaries, ignore_biz = {}, set()
    for split in ("train", "val"):
        df = pd.read_csv(gold_dir / f"{split}.csv", sep="\t")
        gold_summaries[split] = [x for i in range(1, 4) for x in df[f"summ{i}"]]
        ignore_biz.update(set(build_gold(df, output_dir / f"{split}_gold.json")))
    df = pd.read_csv(Path(gold_dir) / f"test.csv", sep="\t")
    ignore_biz.update(build_gold(df, output_dir / f"test.json"))
    return gold_summaries, ignore_biz


def length_filter(data: List[dict],
                  min_len: int,
                  max_len: int):
    with Pool(cpu_count()) as p:
        texts = [x["text"] for x in data]
        tokenized = p.map(nltk.word_tokenize, texts)
        word_len = p.map(len, tokenized)
    for tok, wln, x in zip(tokenized, word_len, data):
        x["text"] = " ".join(tok)
        x["word_len"] = wln
    return [x for x in data if min_len <= x["word_len"] <= max_len]


def lang_filter(data: List[dict], lang: str = "en"):
    texts = [x["text"] for x in data]
    model = fasttext.load_model("lid.176.bin")
    pred = model.predict(texts)[0]
    return [x for p, x in zip(pred, data) if p[0] == f"__label__{lang}"]


def group_by(data: List[dict], min_ent: int = 10, src_min_len: int = 40, src_max_len: int = 70):
    group_by_biz = defaultdict(list)
    for ins in data:
        group_by_biz[ins["business_id"]].append(ins)
    grouped = []

    for vs in group_by_biz.values():
        obs, reviews = set(), []
        # No Duplicated text
        for v in vs:
            if v["text"] not in obs:
                reviews.append(v)
                obs.add(v["text"])
        if sum(src_min_len <= x["word_len"] <= src_max_len for x in reviews) >= min_ent:
            grouped.append(reviews)
    return grouped


def pairing(reviews: List[dict],
            summaries: Dict[str, List[str]],
            v: TfidfVectorizer,
            src_min_len: int = 40,
            src_max_len: int = 70,
            tgt_min_len: int = 40,
            tgt_max_len: int = 70):
    src_rev = [x for x in reviews if src_min_len <= x["word_len"] <= src_max_len]
    src_vec = v.transform([x["text"] for x in src_rev])
    nn = NearestNeighbors(n_neighbors=9).fit(src_vec)
    # Pseudo
    tgt_rev = [x for x in reviews if tgt_min_len <= x["word_len"] <= tgt_max_len]
    tgt_vec = v.transform([x["text"] for x in tgt_rev])
    dist, src_ind = nn.kneighbors(tgt_vec)  # dist: l2 dist of normalized vectors
    dist, src_ind = dist[:, 1:], src_ind[:, 1:]

    pseudo = []
    for ins, indices, d in zip(tgt_rev, src_ind, dist):
        if len(set(d)) != 8:  # Duplicated inputs
            continue
        src = sorted([src_rev[i] for i in indices], key=lambda x: -x["word_len"])
        pseudo.append({
            "src": [x["text"] for x in src],
            "src_id": [x["review_id"] for x in src],
            "tgt": nltk.word_tokenize(ins["text"]),
            "tgt_id": ins["review_id"],
            "avg_dist": d.mean(),
            "dist": d.tolist()
        })

    # Noisy
    noisy = defaultdict(list)
    for split in ("train", "val"):
        tgt_vec = v.transform(summaries[split])
        dist, src_ind = nn.kneighbors(tgt_vec)  # dist: l2 dist of normalized vectors
        dist, src_ind = dist[:, :-1], src_ind[:, :-1]
        for tgt, indices, d in zip(summaries[split], src_ind, dist):
            if len(set(d)) != 8:  # Duplicated inputs
                continue
            src = sorted([src_rev[i] for i in indices], key=lambda x: -x["word_len"])
            noisy[split].append({
                "src": [x["text"] for x in src],
                "src_id": [x["review_id"] for x in src],
                "tgt": nltk.word_tokenize(tgt),
                "tgt_id": "",
                "avg_dist": d.mean(),
                "dist": d.tolist()
            })
    return pseudo, noisy


def build_pseudo_noisy(data_dir: Union[str, Path],
                       data: List[List[dict]],
                       gold_summaries: Dict[str, List[str]],
                       topk: int = 100000,
                       src_min_len: int = 40,
                       src_max_len: int = 70,
                       tgt_min_len: int = 40,
                       tgt_max_len: int = 70):
    print("Building pseudo and noisy set")
    v: TfidfVectorizer = TfidfVectorizer()
    v.fit([x["text"] for xs in data for x in xs])

    with Pool(cpu_count()) as p:
        pseudo, noisy = [], defaultdict(lambda: defaultdict(list))
        func = partial(pairing,
                       v=v,
                       summaries=gold_summaries,
                       src_min_len=src_min_len,
                       src_max_len=src_max_len,
                       tgt_min_len=tgt_min_len,
                       tgt_max_len=tgt_max_len)
        for p_data, n_data in tqdm(p.imap_unordered(func, data), total=len(data), dynamic_ncols=True, desc="Build.."):
            pseudo.extend(p_data)
            for split in ("train", "val"):
                for n in n_data[split]:
                    noisy[split][tuple(n["tgt"])].append(n)

    pseudo = filter(lambda x: min(x["dist"]) >= 0.8, pseudo)  # Remove too similar input-output pairs
    pseudo = sorted(pseudo, key=lambda x: x["avg_dist"])
    pseudo = pseudo[:topk]
    noisy = {split: [x for xs in noisy[split].values() for x in sorted(xs, key=lambda x: x["avg_dist"])[:10]] for split
             in ("train", "val")}
    json.dump(pseudo, open(data_dir / "pseudo.json", "w"))
    json.dump(noisy["train"], open(data_dir / "noisy.json", "w"))
    json.dump(noisy["val"], open(data_dir / "val.json", "w"))
    print("Pseudo size: ", len(pseudo))
    print("Noisy size: ", len(noisy))


def amazon(data_dir,
           min_len: int = 40,
           max_len: int = 70,
           min_ent: int = 10,
           topk: int = 100000):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    gold_summaries, ignore_biz = process_gold(FEWSUM_DIR.format("amazon"), data_dir)

    obs = set()
    raw = []
    for category in ("Clothing_Shoes_and_Jewelry", "Electronics", "Health_and_Personal_Care", "Home_and_Kitchen"):
        for line in tqdm(list(gzip.open(Path(data_dir) / f"reviews_{category}.json.gz", "rb")), desc=category):
            ins = json.loads(line)
            text = strip_text(ins["reviewText"])
            business_id, review_id = ins["asin"], ins["reviewerID"]
            if business_id in ignore_biz:
                continue
            if (business_id, review_id) not in obs:
                raw.append({"business_id": business_id,
                            "review_id": review_id,
                            "text": text})
                obs.add((business_id, review_id))
    raw = length_filter(raw, min_len=min_len, max_len=max_len)
    raw = lang_filter(raw)
    grouped = group_by(raw, min_ent=min_ent)
    build_pseudo_noisy(data_dir, grouped, gold_summaries, topk=topk)


def michelin(data_dir,
             src_min_len: int = 64,
             src_max_len: int = 128,
             tgt_min_len: int = 96,
             tgt_max_len: int = 192,
             min_ent: int = 10,
             topk: int = 100000):
    print("Process Michelin")
    data_dir = Path(data_dir)

    gold_summaries = defaultdict(list)
    for split in ("train", "val", "test"):
        outputs = []
        df = pd.read_json(data_dir / "prosum" / f"{split}.json")
        for _, ins in df.iterrows():
            outputs.append({
                "cat": "michelin",
                "business_id": ins["michelin_url"],
                "src": [strip_text(x) for x in ins["yelp_reviews"]],
                "ref": [unidecode(ins["michelin_pov"])],  # Only convert raw to ascii
                "tgt": nltk.word_tokenize(strip_text(ins["michelin_pov"]))})
            gold_summaries[split].append(strip_text(ins["michelin_pov"]))
        if split != "test":
            split = f"{split}_gold"
        json.dump(outputs, open(data_dir / f"{split}.json", "w"))

    biz = set()
    for line in tqdm(open(data_dir / "yelp_academic_dataset_business.json")):
        ins = json.loads(line)
        if ins["categories"] is None or ins["stars"] < 4.0:
            continue
        cat = ins["categories"].lower()
        if "restaurant" in cat or "food" in cat:
            biz.add(ins["business_id"])

    raw = []
    for line in tqdm(open(data_dir / "yelp_academic_dataset_review.json")):
        ins = json.loads(line)
        if ins["stars"] == 5. and ins["business_id"] in biz:
            ins["text"] = strip_text(ins["text"])
            raw.append(ins)
    raw = length_filter(raw,
                        min_len=min(src_min_len, tgt_min_len),
                        max_len=max(src_max_len, tgt_max_len))
    raw = lang_filter(raw)
    grouped = group_by(raw, min_ent=min_ent, src_min_len=src_min_len, src_max_len=src_max_len)
    build_pseudo_noisy(data_dir,
                       grouped,
                       gold_summaries=gold_summaries,
                       topk=topk,
                       src_min_len=src_min_len,
                       src_max_len=src_max_len,
                       tgt_min_len=tgt_min_len,
                       tgt_max_len=tgt_max_len)


def yelp(data_dir,
         min_len: int = 40,
         max_len: int = 70,
         min_ent: int = 10,
         topk: int = 100000):
    print("Process YELP")
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    gold_summaries, ignore_biz = process_gold(FEWSUM_DIR.format("yelp"), data_dir)

    # Pseudo training data
    raw = []
    for line in tqdm(list(open(data_dir / "yelp_academic_dataset_review.json")), desc="Preprocess", dynamic_ncols=True):
        ins = json.loads(line)
        if ins["business_id"] in ignore_biz:
            continue
        ins["text"] = strip_text(ins["text"])
        raw.append(ins)

    raw = length_filter(raw, min_len=min_len, max_len=max_len)
    raw = lang_filter(raw)
    grouped = group_by(raw, min_ent=min_ent)
    build_pseudo_noisy(data_dir, grouped, gold_summaries, topk=topk)


def run(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    os.system("git clone https://github.com/abrazinskas/FewSum.git --depth 1")  # Download fewsum
    yelp(data_dir / "yelp")
    amazon(data_dir / "amazon")
    shutil.rmtree("FewSum")
    michelin(data_dir / "michelin")


if __name__ == '__main__':
    fire.Fire(run)
