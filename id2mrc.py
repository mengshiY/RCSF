# encoding: utf-8

import re
import os
import json
from utils import bio_decode

domain2slots = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}
domain2desp = {"AddToPlaylist": "add to playlist", "BookRestaurant": "reserve restaurant", "GetWeather": "get weather", "PlayMusic": "play music", "RateBook": "rate book", "SearchCreativeWork": "search creative work", "SearchScreeningEvent": "search screening event"}
slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 'facility': 'facility', 'movie_name': 'movie name', 'location_name': 'location name', 'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 'poi': 'position', 'party_size_description': 'person', 'served_dish': 'served dish', 'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 'album': 'album', 'object_name': 'object name', 'state': 'location', 'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'owner', 'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}

# use two examples
slot2example = {
    # AddToPlaylist
    "music_item": ["song", "track"],
    "playlist_owner": ["my", "donna s"],
    "entity_name": ["the crabfish", "natasha"],
    "playlist": ["quiero playlist", "workday lounge"],
    "artist": ["lady bunny", "lisa dalbello"],
    # BookRestaurant
    "city": ["north lima", "falmouth"],
    "facility": ["smoking room", "indoor"],
    "timeRange": ["9 am", "january the twentieth"],
    "restaurant_name": ["the maisonette", "robinson house"],
    "country": ["dominican republic", "togo"],
    "cuisine": ["ouzeri", "jewish"],
    "restaurant_type": ["tea house", "tavern"],
    "served_dish": ["wings", "cheese fries"],
    "party_size_number": ["seven", "one"],
    "poi": ["east brady", "fairview"],
    "sort": ["top-rated", "highly rated"], 
    "spatial_relation": ["close", "faraway"],
    "state": ["sc", "ut"],
    "party_size_description": ["me and angeline", "my colleague and i"],
    # GetWeather
    "current_location": ["current spot", "here"],
    "geographic_poi": ["bashkirsky nature reserve", "narew national park"],
    "condition_temperature": ["chillier", "hot"],
    "condition_description": ["humidity", "depression"],
    # PlayMusic
    "genre": ["techno", "pop"],
    "service": ["spotify", "groove shark"],
    "year": ["2005", "1993"],
    "album": ["allergic", "secrets on parade"],
    "track": ["in your eyes", "the wizard and i"],
    # RateBook
    "object_part_of_series_type": ["series", "saga"],
    "object_select": ["this", "current"],
    "rating_value": ["1", "four"],
    "object_name": ["american tabloid", "my beloved world"],
    "object_type": ["book", "novel"],
    "rating_unit": ["points", "stars"],
    "best_rating": ["6", "5"],
    # SearchCreativeWork
    # SearchScreeningEvent
    "movie_type": ["animated movies", "films"],
    "object_location_type": ["movie theatre", "cinema"],
    "location_name": ["amc theaters", "wanda group"],
    "movie_name": ["on the beat", "for lovers only"]
}

domain2slots['atis'] = []
with open("data/atis/labels.txt", 'r') as fr:
    for line in fr:
        line_strip = line.strip('\n').split('\t')
        slot = line_strip[0]
        slot2desp[slot] = line_strip[1]
        domain2slots['atis'].append(slot)
        slot2example[slot] = line_strip[2:]
        print(slot2example[slot])

def get_unique_slot():
    domain2uniqueSlots = dict()
    seen_slots = []
    unseen_slots = []
    for domain, slots in domain2slots.items():
        domain2uniqueSlots[domain] = []
        for slot in slots:
            flag = True
            for k, v in domain2slots.items():
                if k == domain:
                    continue
                else:
                    if slot in v:
                        flag = False
                        break
            if flag:
                domain2uniqueSlots[domain].append(slot)
                unseen_slots.append(slot)
            else:
                seen_slots.append(slot)
    return domain2uniqueSlots, seen_slots, unseen_slots

def convert2mrc(data, dm_name, is_train=True, query_type="desp"):
    origin_count = 0
    new_count = 0
    mrc_samples = []
    slot2query = {}

    if query_type == "desp":
        for slot, desp in slot2desp.items():
            slot2query[slot] = "what is the {}".format(desp)
        print("Using queries from description: {}".format(slot2query))
    elif query_type == "trans":
        with open('data/slot2query.txt', 'r') as f:
            for line in f:
                slot, query = line.strip().split("\t")
                slot2query[slot] = query
        print("Using queries from translations: {}".format(slot2query))
    elif query_type == 'example':
        for slot, desp in slot2desp.items():
            slot2query[slot] = "what is the {}".format(desp) + ' like ' + ' or '.join(slot2example[slot])
        print("Using queries from examples: {}".format(slot2query))
        
    for line in data:
        origin_count += 1
        src, labels = line.strip().split("\t")
        tags = bio_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
        # print(tags)
        # exit(0)
        for tag_idx, (label, query) in enumerate(slot2query.items()):
    
            # if len([tag.begin for tag in tags if tag.tag == label]) > 0:
            if label not in domain2slots[dm_name]:
                continue
            mrc_samples.append(
                {
                    "context": ' '.join(src.split()),
                    "start_position": [tag.begin for tag in tags if tag.tag == label],
                    "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                    "query": query,
                    "qas_id": f"{origin_count}.{tag_idx}",
                    "label": label,
                    "tags": labels
                }
            )
            new_count += 1
    return mrc_samples
    # json.dump(mrc_samples, open(output_file, "w"), ensure_ascii=False, sort_keys=True, indent=2)
    # print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")



def run_dataset():
    pass

if __name__ == '__main__':
    # data = ['add sugarolly days to my list  your favorite slaughterhouse	O B-entity_name I-entity_name O B-playlist_owner O B-playlist I-playlist I-playlist',
    #     'add sugarolly days to my list  your	O B-entity_name I-entity_name O B-playlist_owner O B-playlist',
    #     'add slimm cutta calhoun to my this is prince playlist	O B-artist I-artist I-artist O B-playlist_owner B-playlist I-playlist I-playlist O']
    # convert2mrc(data)
    # print(get_unique_slot())
    pass