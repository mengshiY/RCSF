# encoding: utf-8


import argparse
import os
from collections import namedtuple
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, BertForQuestionAnswering
from torch.optim import SGD

# from dataset import MRCNERDataset
from dataset import TruncateDataset
from dataset import collate_to_max_length
from dataset import get_dataloader, get_dataloader_test
from metrics import QuerySpanF1, cal_f1, extract_origin_cal_f1
from model import BertQueryNerConfig, BERTMRC, BERTPretrainedMRC
# from loss import *
from utils import get_parser, set_random_seed, extract_flat_spans_batch
import logging
from id2mrc import slot2desp, domain2slots

set_random_seed(0)

BERTModel = {
    "BERTMRC": BERTMRC,
    "BERTPretrainedMRC": BERTPretrainedMRC
}
BERT_DIR = {
    "BERTMRC": 'bert-base-uncased',
    "BERTPretrainedMRC": 'bert-large-uncased-whole-word-masking-finetuned-squad2'
}


class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        # self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir
        self.bert_config_dir = BERT_DIR[self.args.model]
        self.tokenizer = BertWordPieceTokenizer(vocab=self.bert_config_dir+'/vocab.txt')
        

        if self.args.model == 'BERTMRC':
            self.tokenizer = BertWordPieceTokenizer(vocab=self.bert_config_dir+'/vocab.txt')
            bert_config = BertQueryNerConfig.from_pretrained(self.bert_config_dir,
                                                            hidden_dropout_prob=args.bert_dropout,
                                                            attention_probs_dropout_prob=args.bert_dropout,
                                                            mrc_dropout=args.mrc_dropout)

            self.model = BERTModel[self.args.model].from_pretrained(self.bert_config_dir,
                                                    config=bert_config)
        else:
            # self.tokenizer = AutoTokenizer.from_pretrained(self.bert_config_dir, do_lower_case=True)
            # self.model = BertForQuestionAnswering.from_pretrained(self.bert_config_dir)
            self.model = BERTModel[self.args.model](self.bert_config_dir, self.args)
        logging.info(str(self.model))
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        # self.ce_loss = CrossEntropyLoss(reduction="none")
        self.loss_type = args.loss_type
        # self.loss_type = "bce"
        if self.loss_type == "bce":
            self.bce_loss = BCEWithLogitsLoss(reduction="none")
        else:
            self.dice_loss = DiceLoss(with_logits=True, smooth=args.dice_smooth)
        # todo(yuxian): 由于match loss是n^2的，应该特殊调整一下loss rate
        weight_sum = args.weight_start + args.weight_end + args.weight_span
        self.weight_start = args.weight_start / weight_sum
        self.weight_end = args.weight_end / weight_sum
        self.weight_span = args.weight_span / weight_sum
        self.flat_ner = args.flat
        self.span_f1 = QuerySpanF1(flat=self.flat_ner)
        self.chinese = args.chinese
        self.optimizer = args.optimizer
        self.span_loss_candidates = args.span_loss_candidates
        self.dataset_train, self.dataset_valid, self.dataset_test = get_dataloader(args.tgt_domain, args.n_samples, args.batch_size, self.tokenizer, query_type=self.args.query_type)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="mrc dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.1,
                            help="bert dropout rate")
        parser.add_argument("--weight_start", type=float, default=1.0)
        parser.add_argument("--weight_end", type=float, default=1.0)
        parser.add_argument("--weight_span", type=float, default=1.0)
        parser.add_argument("--flat", action="store_true", help="is flat ner")
        parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "gold"],
                            default="all", help="Candidates used to compute span loss")
        parser.add_argument("--chinese", action="store_true",
                            help="is chinese dataset")
        parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce",
                            help="loss type")
        parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                            help="loss type")
        parser.add_argument("--dice_smooth", type=float, default=1e-8,
                            help="smooth value of dice loss")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """"""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        if self.loss_type == "bce":
            start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
            start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
            end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
            end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            if span_logits is not None:
                match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
                match_loss = match_loss * float_match_label_mask
                match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
            else:
                match_loss = 0
        else:
            start_loss = self.dice_loss(start_logits, start_labels.float(), start_float_label_mask)
            end_loss = self.dice_loss(end_logits, end_labels.float(), end_float_label_mask)
            if span_logits is not None:
                match_loss = self.dice_loss(span_logits, match_labels.float(), float_match_label_mask)
            else:
                match_loss = 0
        return start_loss, end_loss, match_loss

    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx, _ = batch

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)
        

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss
        tf_board_logs[f"match_loss"] = match_loss

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx, appendix = batch

        attention_mask = (tokens != 0).long()
        # if self.args.model == 'BERTMRC':
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)
        # else:
        #     print(start_labels)
        #     print(end_labels)
        #     outputs = self.model(tokens, attention_mask, token_type_ids, start_positions=start_labels, end_positions=end_labels)
        #     print(outputs)

        # print(tokens.size())
        # print(start_logits.size())
        # print(end_logits.size())
        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        output[f"val_loss"] = total_loss
        output[f"start_loss"] = start_loss
        output[f"end_loss"] = end_loss
        output[f"match_loss"] = match_loss

        start_preds, end_preds = start_logits > 0, end_logits > 0
            # exit()
        # print(start_preds)
        # print(end_preds)
        # print(span_logits)
        if span_logits is None:
            span_logits = torch.ones([start_logits.size()[0], start_logits.size()[1], start_logits.size()[1]]).cuda()
        span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                     start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                     match_labels=match_labels)
        output["span_f1_stats"] = span_f1_stats
        
        if self.args.model == 'BERTPretrainedMRC':
            # print(start_logits.detach().cpu().numpy()[0])
            output["start_preds"] = torch.softmax(start_logits, -1).detach().cpu().numpy()
            # print(start_preds.detach().cpu().numpy()[0])
            # print(start_preds)
            output["end_preds"] = torch.softmax(end_logits, -1).detach().cpu().numpy()
            output["start_label_mask"] = start_label_mask
            output["end_label_mask"] = end_label_mask
            # start_preds, end_preds = start_preds > 0.5, end_preds > 0.5
        span_preds = span_logits > 0
        extracted_spans_pred = extract_flat_spans_batch(start_preds.cpu().numpy().tolist(), end_preds.cpu().numpy().tolist(), span_preds.cpu().numpy().tolist(), start_label_mask.cpu().numpy().tolist(), end_label_mask.cpu().numpy().tolist(), top_n=self.args.top_n)
        # extracted_spans_label = extract_flat_spans_batch(start_labels.cpu().numpy().tolist(), end_labels.cpu().numpy().tolist(), match_labels.cpu().numpy().tolist(), [[1 for _ in range(len(start_label_mask[0]))] for _ in range(len(start_label_mask))], [[1 for _ in range(len(start_label_mask[0]))] for _ in range(len(start_label_mask))])
        extracted_spans_label = []
        for match_label in match_labels.cpu().numpy().tolist():
            _span = []
            # print(span_pred)
            # exit()
            for i in range(len(match_label)):
                for j in range(len(match_label[i])):
                    # print(match_label[i][j])
                    if match_label[i][j]:
                        # print("*"*10)
                        _span.append((i, j))
            # if(len(_span)>1):
            #     print("-"*20)
            #     print(_span)
            #     print("-"*20)
            extracted_spans_label.append(_span)
        # print(extracted_spans_label)

        # output['start_preds'] = start_preds.cpu().numpy().tolist()
        # output['end_preds'] = end_preds.cpu().numpy().tolist()
        # output['span_preds'] = span_preds.cpu().numpy().tolist()
        # output['label_idx'] = label_idx.cpu().numpy().tolist()
        output['extracted_spans_pred'] = extracted_spans_pred
        output['extracted_spans_label'] = extracted_spans_label
        output['appendix'] = appendix
        spans_pred_pro = extract_flat_spans_batch(output["start_preds"], output['end_preds'], None, output['start_label_mask'], output['end_label_mask'], appendix, 'BERTPretrainedMRC')
        # extract_origin_cal_f1(spans_pred_pro, extracted_spans_label, appendix, self.args.tgt_domain)
        
        # print(output['label_idx']) 
        # print(extracted_spans_pred)
        # print(extracted_spans_label)
        # print(appendix)
        return output

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        print("span_tp, span_fp, span_fn:")
        print(span_tp, span_fp, span_fn)
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        extracted_spans_preds = []
        extracted_spans_labels = []
        appendixes = []
        for output in outputs:
            extracted_spans_preds.extend(output['extracted_spans_pred'])
            extracted_spans_labels.extend(output['extracted_spans_label'])
            appendixes.extend(output['appendix'])
        # for i in range(len(appendixes)):
        #     print(appendixes[i]["context"])
        #     print(appendixes[i]["label"])
        #     print(extracted_spans_labels[i])
        #     print(extracted_spans_preds[i])
        tensorboard_logs['precision'], tensorboard_logs['recall'], tensorboard_logs['f1'] = cal_f1(extracted_spans_labels, extracted_spans_preds, appendixes)
        for slot, _ in slot2desp.items():
            precision, recall, f1 = cal_f1(extracted_spans_labels, extracted_spans_preds, appendixes, slots=[slot])
            if slot in domain2slots[self.args.tgt_domain]:
                print("*"*3, slot, precision, recall, f1)
            else:
                print(slot, precision, recall, f1)
        if self.args.model == 'BERTPretrainedMRC':
            start_preds = []
            end_preds = []
            start_label_masks = []
            end_label_masks = []
            for output in outputs:
                start_preds.extend(output['start_preds'])
                end_preds.extend(output['end_preds'])
                start_label_masks.extend(output['start_label_mask'])
                end_label_masks.extend(output['end_label_mask'])
            spans_preds_pro = extract_flat_spans_batch(start_preds, end_preds, None, start_label_masks, end_label_masks, appendixes, 'BERTPretrainedMRC')
            tensorboard_logs['precision_pretrain'], tensorboard_logs['recall_pretrain'], tensorboard_logs['f1_pretrain'] = cal_f1(extracted_spans_labels, extracted_spans_preds, appendixes)
            for slot, _ in slot2desp.items():
                precision, recall, f1 = cal_f1(extracted_spans_labels, extracted_spans_preds, appendixes, slots=[slot])
                if slot in domain2slots[self.args.tgt_domain]:
                    print("*"*3, slot, precision, recall, f1)
                else:
                    print(slot, precision, recall, f1)
            tensorboard_logs[f"coach_f1"] = extract_origin_cal_f1(spans_preds_pro, extracted_spans_labels, appendixes, self.args.tgt_domain)
        
        return {'val_loss': avg_loss, 'lg': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_train, "train")
        # return self.get_dataloader("dev", 100)

    def val_dataloader(self):
        return self.get_dataloader(self.dataset_valid, "dev")

    def test_dataloader(self):
        return self.get_dataloader(self.dataset_test, "test")
        # return self.get_dataloader("dev")

    def get_dataloader(self, dataset, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        # json_path = os.path.join(self.data_dir, f"mrc-ner.{prefix}")
        # vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        # dataset = MRCNERDataset(json_path=json_path,
        #                         tokenizer=BertWordPieceTokenizer(vocab_path),
        #                         max_length=self.args.max_length,
        #                         is_chinese=self.chinese,
        #                         pad_to_maxlen=False
        #                         )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader


def run_dataloader():
    """test dataloader"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args.workers = 0
    args.default_root_dir = "train_logs/debug"

    model = BertLabeling(args)
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_config_dir, "vocab.txt"))

    loader = model.train_dataloader()
    for d in loader:
        # print(d)
        input_ids = d[0][0].tolist()
        match_labels = d[-1][0]
        # start_positions, end_positions = torch.where(match_labels > 0)
        # start_positions = start_positions.tolist()
        # end_positions = end_positions.tolist()
        # if not start_positions:
        #     continue
        print("="*20)
        # print(input_ids)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        exit()
        # for start, end in zip(start_positions, end_positions):
        #     print(tokenizer.decode(input_ids[start: end+1]))


def main():
    # run_dataloader()
    """main"""
    # '''
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = BertLabeling(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=2,
        verbose=True,
        monitor="coach_f1",
        period=-1,
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="coach_f1",
        patience=args.early_stop,
        verbose=True,
        mode="max",
        min_delta=0.00
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback]
    )

    if not args.only_test:
        trainer.fit(model)
        print(checkpoint_callback.best_model_path)

        # test
        model = BertLabeling.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            map_location=None,
            batch_size=16,
            max_length=128,
            workers=0
        )
    trainer.test(model=model)
    # test on seen and unseen
    print("**********testing on unseen data**********")
    dataset_seen, dataset_unseen = get_dataloader_test(model.args.tgt_domain, tokenizer=model.tokenizer)
    model.dataset_test = dataset_unseen
    trainer.test(model=model)
    print("**********testing on unseen data**********")
    model.dataset_test = dataset_seen
    trainer.test(model=model)



if __name__ == '__main__':
    # run_dataloader()
    main()
