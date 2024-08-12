import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from summarizer import Summarizer, load_data

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Summarizer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--test_path", type=str, )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output_path", type=str, )

    args = parser.parse_args()

    summarizer = Summarizer(**vars(args))
    test = load_data(summarizer.tokenizer, args.test_path)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, )
    dataloader = DataLoader(test, collate_fn=DataCollatorForSeq2Seq(tokenizer=summarizer.tokenizer))
    outputs = trainer.predict(model=summarizer, dataloaders=dataloader, ckpt_path=args.ckpt)
    if args.output_path is not None:
        with open(args.output_path, "w") as file:
            print("\n".join([o for os in outputs for o in os]), file=file)
