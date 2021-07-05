# encoding: utf-8
import re
import os
import json
import torch
import argparse
import numpy as np
from typing import Tuple, List

domain2slots = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}
domain2slots['atis'] = []
domain2desp = {"AddToPlaylist": "add to playlist", "BookRestaurant": "reserve restaurant", "GetWeather": "get weather", "PlayMusic": "play music", "RateBook": "rate book", "SearchCreativeWork": "search creative work", "SearchScreeningEvent": "search screening event"}
slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 'facility': 'facility', 'movie_name': 'moive name', 'location_name': 'location name', 'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 'poi': 'position', 'party_size_description': 'person', 'served_dish': 'served dish', 'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 'album': 'album', 'object_name': 'object name', 'state': 'location', 'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'owner', 'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}
with open("data/atis/labels.txt", 'r') as fr:
    for line in fr:
        slot, desp = line.strip('\n').split('\t')[:2]
        slot2desp[slot] = desp
        domain2slots['atis'].append(slot)

def get_parser():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--data_dir", type=str, default="data/snips/", help="data dir")
    parser.add_argument("--model", type=str, default="BERTMRC", help="model")
    parser.add_argument("--tgt_domain", type=str, help="target_domain")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # few shot learning
    parser.add_argument("--n_samples", type=int, default=50, help="number of samples for few shot learning")

    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--workers", type=int, default=4, help="num workers for dataloader")
    parser.add_argument("--early_stop", type=int, default=5, help="patience for early stop")
    parser.add_argument("--only_test", action="store_true", help="only test the model without training")
    parser.add_argument("--query_type", choices=["desp", "trans", "example"], default="trans", help="way to construct queries")
    parser.add_argument("--load_pretrainedBERT", action="store_true", help="only test the model without training")
    parser.add_argument("--top_n", type=int, default=5)
    
    return parser

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})

def bio_decode(char_label_list):
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "S-LOC")]
        >>> bmes_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
    """
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]


        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end < length and char_label_list[end][1][0] == "I":
                end += 1
            entity = " ".join(char_label_list[i][0] for i in range(idx, end))
            tags.append(Tag(entity, label[2:], idx, end))
            idx = end
            continue
        else:
            print(char_label_list)
            raise Exception("Invalid Inputs")
    for tag in tags:
        if tag.tag not in slot2desp.keys():
            print(tags)
            exit(0)
    return tags

def get_unique_slot(domain2slot, domain):
    seen_slots = []
    unseen_slots = []
    for slot in domain2slot[domain]:
        flag = True
        for k, v in domain2slot.items():
            if k == domain:
                continue
            else:
                if slot in v:
                    flag = False
                    break
        if flag:
            unseen_slots.append(slot)
        else:
            seen_slots.append(slot)
    return seen_slots, unseen_slots

def convert_file(input_file, output_file):
    origin_count = 0
    new_count = 0
    mrc_samples = []
    with open(input_file) as fin:
        for line in fin:
            origin_count += 1
            src, labels = line.strip().split("\t")
            tags = bio_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
            for label, query in slot2desp.items():
                mrc_samples.append(
                    {
                        "context": src,
                        "start_position": [tag.begin for tag in tags if tag.tag == label],
                        "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                        "query": query
                    }
                )
                new_count += 1
    json.dump(mrc_samples, open(output_file, "w"), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")

def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    # print("remove overlap")
    # pri/nt("Before: {}".format(spans))
    output = []
    occupied = set()
    for span in spans:
        # print(start)
        # print(end)
        # print(occupied)
        if len(span) == 2:
            start, end = span
        else:
            slot, (start, end), _, p = span
        is_occupied = False
        for x in range(start, end+1):
            if x in occupied:
                is_occupied = True
                break
        if is_occupied: continue
        # if any(x for x in range(start, end+1)) in occupied:
        #     continue
        output.append(span)
        for x in range(start, end + 1):
            occupied.add(x)
    # if len(output) > 0: 
    #     print("Before: {}".format(spans))
    #     print("After: {}".format(output))
    return output

def extract_flat_spans(start_pred, end_pred, match_pred, start_label_mask, end_label_mask):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_tag = "TAG"
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and start_label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and end_label_mask[idx]]
    # print(start_pred)
    # print(end_pred)
    # print(start_positions)
    # print(end_positions)
    # print(match_pred)

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    # for end_item in end_positions:
    #     bmes_labels[end_item] = f"I-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end+1):
                    bmes_labels[i] = f"I-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"B-{pseudo_tag}"

    tags = bio_decode([(pseudo_input, label) for label in bmes_labels])

    return [(tag.begin, tag.end) for tag in tags]

def extract_flat_spans_1(start_pred, end_pred, match_pred, start_label_mask, end_label_mask):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    spans = []
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and start_label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and end_label_mask[idx]]

    for tmp_start in start_positions:
        tmp_ends = [tmp for tmp in end_positions if tmp >= tmp_start]
        for tmp_end in tmp_ends:
            if match_pred[tmp_start][tmp_end]:
                spans.append((tmp_start, tmp_end))

    # if len(spans) > 0:
    #     print("+"*10)
    #     print(spans)
    #     print("+"*10)
    return remove_overlap(spans)

def extract_flat_spans_by_probability(start_pred, end_pred, start_label_mask, end_label_mask, appendix, top_n=5):
    spans_pro = []
    start_sorted = np.argsort(-start_pred)
    end_sorted = np.argsort(-end_pred)

    cls_indicator = start_pred[0] + end_pred[0]
    
    for start in start_sorted[:top_n]:
        if start_label_mask[start] == 0:
            break
        for end in end_sorted[:top_n]:
            if end_label_mask[end] == 0:
                break
            if start <= end and end - start < 8:
                spans_pro.append((start, end, start_pred[start] + end_pred[end]))
    spans = [(start, end) for start, end, _ in spans_pro]
    return remove_overlap(spans), spans_pro

def extract_flat_spans_batch(start_preds, end_preds, match_preds, start_label_masks, end_label_masks, appendixes="None", model="None", top_n=5):
    pred_spans = []
    for i in range(len(start_preds)):
        # _spans = extract_flat_spans(start_preds[i], end_preds[i], match_preds[i], start_label_masks[i], end_label_masks[i])
        # _spans_1 = extract_flat_spans_1(start_preds[i], end_preds[i], match_preds[i], start_label_masks[i], end_label_masks[i])
        # assert all(span[0] == _spans_1[idx][0] and span[1] == _spans_1[idx][1] for idx, span in enumerate(_spans))
        # if len(_spans) != len(_spans_1):
        #     print("$$$$"*10)
        #     print(_spans)
        #     print(_spans_1)
        if model == 'BERTPretrainedMRC':
            spans_pred, spans_pred_pro = extract_flat_spans_by_probability(start_preds[i], end_preds[i], start_label_masks[i], end_label_masks[i], appendixes[i], top_n=top_n)
            pred_spans.append(spans_pred_pro)
        else:
            pred_spans.append(extract_flat_spans_1(start_preds[i], end_preds[i], match_preds[i], start_label_masks[i], end_label_masks[i]))
    return pred_spans

