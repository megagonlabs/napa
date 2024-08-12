import json
from collections import defaultdict
from pathlib import Path

import fire
import nltk
import pandas as pd
import rouge
import torch
from bert_score import BERTScorer

from ctc import CTC


@torch.no_grad()
def run(hyp_file, test_file):
    hyp_file = Path(hyp_file)
    if hyp_file.is_dir():
        hyp_file = next(hyp_file.glob("test*.hypo"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyp = [x.strip() for x in open(hyp_file)]
    test = json.load(open(test_file))
    ref = [[r.strip() for r in x["ref"]] for x in test]
    src = [x["src"] for x in test]

    print("ROUGE")
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    scores = evaluator.get_scores(hyp, ref).items()
    scores = {"_".join((metric, k)): v for metric, vs in scores for k, v in vs.items()}

    print("BERTScore")
    bert_scorer = BERTScorer(model_type="roberta-large", lang="en", device=device)
    for key, val in zip("prf", bert_scorer.score(hyp, ref)):
        scores[f"bert_score-roberta_{key}"] = val.mean().item()

    novel_ngrams = defaultdict(list)
    for h, s in zip(hyp, src):
        tokenized_src = [nltk.word_tokenize(x.lower()) for x in s]
        tokenized_hyp = nltk.word_tokenize(h.lower())
        for n in range(1, 5):
            src_ngrams = {ng for x in tokenized_src for ng in nltk.ngrams(x, n=n)}
            hyp_ngrams = list(nltk.ngrams(tokenized_hyp, n=n))
            novel_ngram_ratio = 100 * sum(ng not in src_ngrams for ng in hyp_ngrams) / len(hyp_ngrams)
            novel_ngrams[f"novel-{n}gram"].append(novel_ngram_ratio)
    for n in range(1, 5):
        scores[f"novel-{n}gram"] = sum(novel_ngrams[f"novel-{n}gram"]) / len(novel_ngrams[f"novel-{n}gram"])

    ctc = CTC()
    consistency, relevance = zip(*(ctc(s, h, r) for s, h, r in zip(src, hyp, ref)))
    scores["consistency"] = sum(consistency) / len(consistency)
    scores["relevance"] = sum(relevance) / len(relevance)

    scores = pd.Series(scores)
    print(scores)
    scores.to_json(str(hyp_file) + "_score.json")


if __name__ == '__main__':
    fire.Fire(run)
