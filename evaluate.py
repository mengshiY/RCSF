# encoding: utf-8


import os
from pytorch_lightning import Trainer

from trainer import BertLabeling


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[2, 3], distributed_backend="ddp")

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=16,
        max_length=128,
        workers=0
    )
    trainer.test(model=model)


if __name__ == '__main__':
    
    CHECKPOINTS = ''
    HPARAMS = ""


    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
