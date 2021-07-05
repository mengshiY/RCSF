import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
from tokenizers import BertWordPieceTokenizer
from id2mrc import convert2mrc

import logging
logger = logging.getLogger()

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
# domain_set = ["AddToPlaylist", "PlayMusic"]


class MRCSFDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, all_data, tokenizer: BertWordPieceTokenizer, max_length: int = 128, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False):
        self.all_data = all_data
        self.tokenzier = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        # print(query)
        # print(context)
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        # print('start_positions:{}'.format(start_positions))
        # print('end_positions:{}'.format(end_positions))

        if self.is_chinese:
            context = "".join(context.split())
            end_positions = [x+1 for x in end_positions]
        else:
            # add space offsets
            words = context.split()
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        # print('start_positions:{}'.format(start_positions))
        # print('end_positions:{}'.format(end_positions))
        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        # print(query_context_tokens)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets
        # print('tokens:{}'.format(tokens))
        # print('type_ids:{}'.format(type_ids))
        # print('offsets:{}'.format(offsets))
        # exit()


        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        orig_to_tok_index = []
        tok_to_orig_index = []
        token_starts = []
        _start = 0
        for word in context.split():
            token_starts.append(_start)
            _start += (len(word) + 1)
        
        for token_idx in range(len(tokens)):
            tok_to_orig_index.append(0)
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx
            for i, _start in enumerate(token_starts):
                if _start == token_start:
                    tok_to_orig_index[token_idx] = i
                    break
                elif _start > token_start:
                    tok_to_orig_index[token_idx] = i-1
                    break
                elif i == len(token_starts) - 1:
                    tok_to_orig_index[token_idx] = i

                
        
        try:
            orig_to_tok_index = [origin_offset2token_idx_start[i] for i in token_starts]
        except:
            print(context)
            exit()
        # print(orig_to_tok_index)
        # print(tok_to_orig_index)
        
        # exit()

        # start_positions = data["start_position"]
        # end_positions = data["end_position"]
        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
        # print('new_start_positions:{}'.format(new_start_positions))
        # print('new_end_positions:{}'.format(new_end_positions))
        # # exit(0)

        start_positions = data["start_position"]
        end_positions = data["end_position"]
        
        new_start_positions_1 = [orig_to_tok_index[start] for start in start_positions]
        
        # print('new_start_positions:{}'.format(new_start_positions_1))
        assert all(new_start_positions[i] == new_start_positions_1[i] for i in range(len(new_start_positions)))
        # print('new_end_positions:{}'.format(new_end_positions))

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        if not self.is_chinese:
            for token_idx in range(len(tokens)):
                current_word_idx = query_context_tokens.words[token_idx]
                next_word_idx = query_context_tokens.words[token_idx+1] if token_idx+1 < len(tokens) else None
                prev_word_idx = query_context_tokens.words[token_idx-1] if token_idx-1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]
        # label_mask = label_mask[: self.max_length]

        # make sure last token is [SEP]
        # sep_token = 102
        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0
            # label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)
            # label_mask = self.pad(label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        appendix = {
                        "sample_id": qas_id.split(".")[0], 
                        "query": query, 
                        "context": context, 
                        "label": data["label"], 
                        "tok_to_orig_index": tok_to_orig_index,
                        "tags": data['tags']
                    }
        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            # torch.LongTensor(label_mask),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            label_idx,
            appendix
        ]


    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def read_file(filepath):
    sample_list= []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()  # text \t label
            sample_list.append(line)
    print("Loading data from {}, total nums is {}".format(filepath, len(sample_list)))
    return sample_list


def dataReader():
    print("Loading and processing data ...")

    data = {}

    # load data
    for domain in domain_set:
        data[domain] = read_file("data/snips/"+domain+'/'+domain+'.txt')

    data['atis'] = read_file("data/atis/atis.txt")
    
    return data

def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[7] for x in batch]))
    output.append(torch.stack([x[8] for x in batch]))
    output.append([x[9] for x in batch])

    return output

def get_dataloader_test(tgt_domain, batch_size=32, tokenizer=None, query_type="desp"):
    test_seen_org = read_file("data/snips/"+tgt_domain+'/'+'seen_slots.txt')
    test_unseen_org = read_file("data/snips/"+tgt_domain+'/'+'unseen_slots.txt')
    test_seen = convert2mrc(test_seen_org, tgt_domain, query_type=query_type)
    test_unseen = convert2mrc(test_unseen_org, tgt_domain, query_type=query_type)
   
    dataset_seen = MRCSFDataset(test_seen, tokenizer)
    dataset_unseen = MRCSFDataset(test_unseen, tokenizer)
    print("{} is the target domain\ntest_seen: {} -> {}\ntest_unseen: {} -> {}\n".format(tgt_domain, \
        len(test_seen_org), len(dataset_seen), len(test_unseen_org), len(dataset_unseen)))
    
    return dataset_seen, dataset_unseen

def get_dataloader(tgt_domain, n_samples, batch_size=32, tokenizer=None, query_type="desp"):
    all_data = dataReader()
    train_data = []
    valid_data = []
    test_data = []
    train_data_orig = []
    valid_data_orig = []
    test_data_orig = []
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain and dm_name != 'atis':
            train_data.extend(convert2mrc(dm_data, dm_name, query_type=query_type))
            train_data_orig.extend(dm_data)
            print("{}: {} -> {}".format(dm_name, len(dm_data), len(convert2mrc(dm_data, dm_name, query_type=query_type))))
    
    train_data.extend(convert2mrc(all_data[tgt_domain][:n_samples], tgt_domain, query_type=query_type))
    train_data_orig.extend(all_data[tgt_domain][:n_samples])
    valid_data = convert2mrc(all_data[tgt_domain][n_samples:500], tgt_domain, query_type=query_type)
    # print(valid_data)
    valid_data_orig.extend(all_data[tgt_domain][n_samples:500])
    # print("OK")
    test_data = convert2mrc(all_data[tgt_domain][500:], tgt_domain, query_type=query_type)
    test_data_orig.extend(all_data[tgt_domain][500:])

    dataset_train = MRCSFDataset(train_data, tokenizer)
    dataset_valid = MRCSFDataset(valid_data, tokenizer)
    dataset_test = MRCSFDataset(test_data, tokenizer)
    print("{} is the target domain\nn_samples:{}\ntrain: {} -> {}\nvalid: {} -> {}\ntest: {} -> {}".format(tgt_domain, \
        n_samples, len(train_data_orig), len(dataset_train), len(valid_data_orig), len(dataset_valid), len(test_data_orig), len(dataset_test)))
    
    # train_dataloader = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_to_max_length)
    # valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, collate_fn=collate_to_max_length)
    # test_dataloader = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate_to_max_length)
    # for batch in train_dataloader:
    #     print(batch)
    #     exit()
    # return train_dataloader, valid_dataloader, test_dataloader
    return dataset_train, dataset_valid, dataset_test

class TruncateDataset(Dataset):
    """Truncate dataset to certain num"""
    def __init__(self, dataset: Dataset, max_num: int = 100):
        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))

    def __len__(self):
        return self.max_num

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.dataset, item)


def run_dataset():
    get_dataloader("PlayMusic", 10)


if __name__ == '__main__':
    run_dataset()