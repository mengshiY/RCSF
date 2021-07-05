import torch
from pytorch_lightning.metrics.metric import TensorMetric
from utils import bio_decode, domain2slots, get_unique_slot, remove_overlap
from conll2002_metrics import *

def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    # print(start_label_mask)
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])

def cal_f1(extracted_spans_labels, extracted_spans_preds, appendixes, slots=None):
    n_gold = 0
    n_pred = 0
    n_correct = 0
    # print("*"*10)
    # print(len(extracted_spans_preds))
    # print("*"*10)
    for i in range(len(extracted_spans_preds)):
        if slots is not None and appendixes[i]['label'] not in slots:
            continue
        l = extracted_spans_labels[i]
        pl = extracted_spans_preds[i]
        n_gold += len(l)
        n_pred += len(pl)
        for e in l:
            for t in pl:
                if e[0] == t[0] and e[1] == t[1]:
                    # print(appendixes[i]['context'])
                    # print(appendixes[i]['label'])
                    # print(f"{appendixes[i]['tok_to_orig_index'][e[0]]}-{appendixes[i]['tok_to_orig_index'][e[1]]}")
                    # print(f"{appendixes[i]['tok_to_orig_index'][t[0]]}-{appendixes[i]['tok_to_orig_index'][t[1]]}")
                    # print(t)
                    n_correct += 1
                    break
    precision = n_correct / (n_pred + 1e-10)
    recall = n_correct / (n_gold + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    # print("*"*10)
    print("n_gold, n_pred, n_correct:{}\t{}\t{}".format(n_gold, n_pred, n_correct))
    # print("*"*10)
    return precision, recall, f1

def convert_span_to_tags(span, context):
    # print(appendix)
    # print(span)
    tags = ['O'] * len(context.split())
    span = sorted(span, key=lambda x: x[1][1])
    # print(span)
    for slot, (start, end), appendix, p in span:
        tok_to_orig_index = appendix['tok_to_orig_index']
        # try:
        orig_start = tok_to_orig_index[start]
        orig_end = tok_to_orig_index[end]
        if orig_start-1 >= 0 and tags[orig_start-1][2:] == slot:
            tags[orig_start] = 'I-' + slot
            print("*"*10 + "Found Same!!!" + "*"*10)
        else:
            tags[orig_start] = 'B-' + slot
        for idx in range(orig_start+1, orig_end+1):
            tags[idx] = 'I-' + slot
        # except:
        #     print(appendix)
        #     print(span)

        #     print(slot, (start, end), p)
            # exit()
    return tags

def extract_origin_cal_f1(pred_spans, label_spans, appendixes, tgt_dm):
    lines = []
    slots_lines = {}
    seen_lines = []
    unseen_lines = []
    t_pred = {}
    t_gold = {}
    contexts = {}
    gold_tags = {}
    flag = False
    for slot in domain2slots[tgt_dm]:
        slots_lines[slot] = []
    assert len(pred_spans) == len(label_spans) == len(appendixes)
    for pred_span, label_span, appendix in zip(pred_spans, label_spans, appendixes):
        idx = appendix['sample_id']
        # if idx == '1319':
        #     print(appendix)
        #     print(pred_span)
        #     print(label_span)
        #     flag = True
        t_pred.setdefault(idx, list())
        t_gold.setdefault(idx, list())
        for start, end, p in pred_span:
            t_pred[idx].append([appendix['label'], (start, end), appendix, p])
        for start, end in label_span:
            t_gold[idx].append([appendix['label'], (start, end), appendix, 1])
        contexts[idx] = appendix['context']
        gold_tags[idx] = appendix['tags']
    # if flag:
    #     exit()
    for idx in t_pred:
        elem_pred = t_pred[idx]
        elem_pred = sorted(elem_pred, key=lambda x: x[-1], reverse=True)
        elem_pred = remove_overlap(elem_pred)
        # print(elem_pred)
        # print(remove_overlap(elem_pred))
        elem_gold = t_gold[idx]
        context = contexts[idx]
    # for pred_span, label_span, appendix in zip(pred_spans, label_spans, appendixes):
        # pred_span
        pred_tags = convert_span_to_tags(elem_pred, context)
        label_tags = convert_span_to_tags(elem_gold, context)
        # print(elem_gold)
        if ' '.join(label_tags) != gold_tags[idx]:
            print(elem_gold)
            print(label_tags)
            print(' '.join(label_tags))
            print(gold_tags[idx])
        # assert ' '.join(label_tags) == gold_tags[idx]
        # print(appendix['context'])
        # print(pred_tags)
        # print(label_tags)
        # if len(context.split()) != len(pred_tags):
        #     print(context)
        #     print(pred_tags)
        assert len(context.split()) == len(pred_tags) == len(label_tags)
        for w, pred, gold in zip(context.split(), pred_tags, label_tags):
            lines.append(w + ' ' + pred + ' ' + gold)
            seen_slots, unseen_slots = get_unique_slot(domain2slots, tgt_dm)
            if pred != 'O' and pred[2:] in seen_slots:
                pred_seen = pred
            else:
                pred_seen = 'O'
            if gold != 'O' and gold[2:] in seen_slots:
                gold_seen = gold
            else:
                gold_seen = 'O'
            seen_lines.append("w" + " " + pred_seen + " " + gold_seen)

            if pred != 'O' and pred[2:] in unseen_slots:
                # print(pred)
                pred_unseen = pred
            else:
                pred_unseen = 'O'
            if gold != 'O' and gold[2:] in unseen_slots:
                gold_unseen = gold
            else:
                gold_unseen = 'O'
            unseen_lines.append("w" + " " + pred_unseen + " " + gold_unseen)

            for slot in domain2slots[tgt_dm]:
                slot_types = ('B-'+slot, 'I-'+slot)
                if pred in slot_types:
                    _pred = pred
                else:
                    _pred = 'O'
                if gold in slot_types:
                    _gold = gold
                else:
                    _gold = 'O'
                # if pred in slot_types:
                #     print(pred)
                #     print(slots_lines[slot])
                slots_lines[slot].append("w" + " " + _pred + " " + _gold)
    for slot in domain2slots[tgt_dm]:
        slot_result = conll2002_measure(slots_lines[slot])
        print("Evaluation on slot: {}, f1: {:.4f}".format(slot, slot_result['fb1']))
    
    seen_result = conll2002_measure(seen_lines)
    unseen_result = conll2002_measure(unseen_lines)
    print("Evaluation on unseen slots: {}, f1: {:.4f}".format(unseen_slots, unseen_result['fb1']))
    print("Evaluation on seen slots: {}, f1: {:.4f}".format(seen_slots, seen_result['fb1']))
    result = conll2002_measure(lines)
    f1_score = result["fb1"]
    print("Eval on test set. Slot F1 Score: {:.4f}.".format(f1_score))
    return f1_score


class QuerySpanF1(TensorMetric):
    """
    Query Span F1
    Args:
        flat: is flat-ner
    """
    def __init__(self, reduce_group=None, reduce_op=None, flat=False):
        super(QuerySpanF1, self).__init__(name="query_span_f1",
                                          reduce_group=reduce_group,
                                          reduce_op=reduce_op)
        self.flat = flat

    def forward(self, start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
        return query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels,
                             flat=self.flat)
