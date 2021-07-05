# encoding: utf-8


import os
from pytorch_lightning import Trainer

from trainer import BertLabeling
from dataset import get_dataloader_test


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[1])

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=16,
        max_length=128,
        workers=0
    )
    dataset_seen, dataset_unseen = get_dataloader_test(model.args.tgt_domain, tokenizer=model.tokenizer)
    model.dataset_test = dataset_unseen
    trainer.test(model=model)
    model.dataset_test = dataset_seen
    trainer.test(model=model)


if __name__ == '__main__':
    
    CHECKPOINTS = ""
    HPARAMS = ""
    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
