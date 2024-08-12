import json
import re
from pathlib import Path
from typing import Optional, Union

import bert_score
import pytorch_lightning as pl
import rouge
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamW, \
    get_linear_schedule_with_warmup, set_seed
from transformers.trainer_pt_utils import LabelSmoother

from ctc import CTC

SCORER = bert_score.BERTScorer(model_type="roberta-large", lang="en")


def load_data(tokenizer,
              data_path: Union[str, Path]):
    model_inputs = []
    for ins in json.load(open(data_path)):
        inp = tokenizer(" | ".join(ins["src"]), truncation=True)
        with tokenizer.as_target_tokenizer():
            target = labels = tokenizer(ins["tgt"], is_split_into_words=True, truncation=True).input_ids
            inp["target"] = target
            if "non_hallucinated_mask" in ins:
                mask = ins["non_hallucinated_mask"]
                labels = [lbl if m else -100 for lbl, m in zip(labels, mask)]  # Ignore hallucinated tokens for training
            inp["labels"] = labels
            if "ref" in ins:
                inp["ref"] = tokenizer(" | ".join(ins["ref"])).input_ids
        model_inputs.append(inp)
    return model_inputs


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer,
                 train_path: str,
                 val_path: str = None,
                 **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_path = Path(train_path)
        self.data_dir = self.train_path.parent
        self.val_path = val_path if val_path is not None else self.data_dir / "val.json"
        self.data = {}

    def setup(self, stage: Optional[str] = None) -> None:
        train = load_data(self.tokenizer, self.train_path)
        val = load_data(self.tokenizer, self.data_dir / "val.json")
        test = load_data(self.tokenizer, self.data_dir / "test.json")
        self.data.update({"train": train, "val": val, "test": test})

    def _get_dataloader(self, dataset_split, is_train: bool = False) -> DataLoader:
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        return DataLoader(dataset_split, collate_fn=data_collator, shuffle=is_train)

    def train_dataloader(self):
        return self._get_dataloader(self.data["train"], is_train=True)

    def val_dataloader(self):
        return self._get_dataloader(self.data["val"])

    def test_dataloader(self):
        return self._get_dataloader(self.data["test"])

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--train_path", type=str)
        return parent_parser


class Summarizer(pl.LightningModule):
    def __init__(self,
                 model_name: str = "facebook/bart-large",
                 max_output_len: int = 256,
                 lr: float = 1e-5,
                 weight_decay: float = 0.001,
                 max_steps: int = 50000,
                 warmup: int = 1000,
                 epsilon: float = 0.1,
                 **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.label_smoother = LabelSmoother(epsilon=epsilon) if epsilon > 0. else None

        self.rouge = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                                 stemming=True, ensure_compatibility=True)
        self.ctc = CTC()
        self.max_output_len = max_output_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup = warmup

        self.save_hyperparameters()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs):
        return self.model(input_ids,
                          attention_mask=attention_mask,
                          labels=labels, use_cache=False)

    def training_step(self, batch, batch_nb):
        outputs = self.forward(**batch)
        if self.label_smoother is None:
            loss = outputs.loss
        else:
            loss = self.label_smoother(outputs, batch["labels"])
        self.log("loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup,
                                                    num_training_steps=self.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def predict_step(self, batch, batch_nb, dataloader_idx: int = None):
        generated_ids = self.model.generate(batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            min_length=32, use_cache=True, max_length=self.max_output_len,
                                            num_beams=4, no_repeat_ngram_size=3)
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        return predictions

    @torch.no_grad()
    def _evaluation_step(self, batch, batch_nb):
        sources = self.tokenizer.batch_decode(batch["input_ids"].tolist(), skip_special_tokens=True)
        predictions = self.predict_step(batch, batch_nb)
        if "ref" in batch:
            references = self.tokenizer.batch_decode(batch["ref"], skip_special_tokens=True)
        else:
            print("No ref key")
            references = self.tokenizer.batch_decode(batch["target"].tolist(), skip_special_tokens=True)
        return [{"prediction": pred, "reference": ref, "source": src} for pred, ref, src in
                zip(predictions, references, sources)]

    def validation_step(self, batch, batch_nb):
        return self._evaluation_step(batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self._evaluation_step(batch, batch_nb)

    def _evaluation_epoch_end(self, split, outputs):
        src = [o["source"].strip().split(" | ") for outs in outputs for o in outs]
        hyp = [o["prediction"] for outs in outputs for o in outs]
        ref = [o["reference"].split(" | ") for outs in outputs for o in outs]

        consistency, relevance = zip(*(self.ctc(s, h, r) for s, h, r in zip(src, hyp, ref)))
        consistency = sum(consistency) / len(consistency)
        relevance = sum(relevance) / len(relevance)
        self.log(f"{split}_consistency", consistency, on_epoch=True)
        self.log(f"{split}_relevance", relevance, on_epoch=True)

        scores = {}
        results = self.rouge.get_scores(hyp, ref)
        for metric_name in ("rouge-1", "rouge-2", "rouge-l"):
            for key in "fpr":
                val = results[metric_name][key]
                name = f"{metric_name}/{key}"
                self.log(f"{split}_" + name, val, on_epoch=True, prog_bar=key == "f")
                scores[name] = val
        for key in "fpr":
            val = sum(results[metric_name][key] for metric_name in ("rouge-1", "rouge-2", "rouge-l"))
            self.log(f"{split}_rouge-12l/{key}", val, on_epoch=True)
            if split == "val" and key == "f":
                if self.trainer.global_step <= self.warmup:  # Burn-in
                    self.log("hp_metric", 0.)
                else:
                    self.log("hp_metric", val)
        for key, val in zip("prf", SCORER.score(hyp, ref)):
            val = val.mean().item()
            scores[f"{split}_bertscore/{key}"] = val
            self.log(f"{split}_bertscore/{key}", val)

        if self.trainer.global_step:
            with open(Path(self.trainer.log_dir) / f"{split}_{self.trainer.global_step}.hypo", "w") as file:
                print("\n".join(hyp), file=file)
            text_to_log = ""
            for h, r in zip(hyp[:10], ref[:10]):
                text_to_log += "### Hypothesis\n"
                text_to_log += f"- {h}\n"
                text_to_log += "### Reference\n"
                text_to_log += f"- {r}\n\n"
            self.logger.experiment.add_text(split, text_to_log, self.trainer.global_step)

    def validation_epoch_end(self, outputs):
        return self._evaluation_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self._evaluation_epoch_end("test", outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Summarizer")
        parser.add_argument("--model_name", type=str, default="facebook/bart-large")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--weight_decay", type=float, default=0.001)
        parser.add_argument("--warmup", type=int, default=1000)
        parser.add_argument("--epsilon", type=float, default=0.1)
        return parent_parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Summarizer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=3150)
    parser.add_argument("--ckpt", type=str, default=None)

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    if args.ckpt is None:
        summarizer = Summarizer(**vars(args))
    else:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_dir():
            path, score = None, 0
            for fp in sorted(Path(ckpt_path).iterdir()):
                try:
                    fp = next(fp.iterdir())
                    num = float(re.search(r"f=0.(\d+).ckpt", fp.name).group(1))
                    if num >= score:
                        path = fp
                        score = num
                except AttributeError:
                    print("REGEX Error: ", fp)
                    raise ValueError
            ckpt_path = path
        summarizer = Summarizer.load_from_checkpoint(str(ckpt_path), **vars(args))
    datamodule = DataModule(tokenizer=summarizer.tokenizer, **vars(args))

    checkpoint_callback = ModelCheckpoint(monitor="hp_metric",
                                          verbose=True,
                                          save_top_k=1,
                                          mode="max",
                                          filename="{epoch}-{hp_metric:.4f}")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(summarizer, datamodule=datamodule)
    trainer.test(datamodule=datamodule)
