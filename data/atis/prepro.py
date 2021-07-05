import re
import json
from collections import Counter

def preprocess(filename):
    data = []
    words = []
    labels = []
    slot_name = set()
    with open(filename, 'r') as fr:
        for line in fr:
            tmp = line.strip('\n').split()
            if len(tmp) == 2:
                words.append(tmp[0])
                labels.append(tmp[1])
            elif len(tmp) == 0:
                data.append((' '.join(words), ' '.join(labels)))
                slot_name = slot_name | set([label.replace("B-", "").replace("I-", "") for label in labels])
                words = []
                labels = []
            else:
                print(line)
    return data, slot_name

train_data, slot_train = preprocess('train.txt')
dev_data, slot_dev = preprocess('dev.txt')
test_data, slot_test = preprocess('test.txt')
all_data = train_data + dev_data + test_data
slots = slot_train | slot_dev | slot_test
print(slots)
print(len(slots))
slot2example = {}
slot2allExample = {}
for slot in slots:
    slot2example[slot] = []
    slot2allExample[slot] = []
with open('atis.txt', 'w') as fw:
    for idx, (context, label) in enumerate(all_data):
        # if idx < 500:
        i = 0
        # print(label)
        label_split = label.split()
        while i < len(label_split):
            # print(i)
            l = label_split[i]
            # print(l[0])
            # exit()
            if l[0] == 'O':
                i += 1
                continue
            elif l[0] == 'B':
                end = i + 1
                while end < len(label_split) and label_split[end][0] == "I":
                    end += 1
                entity = " ".join(context.split()[i] for i in range(i, end))
                print(entity)
                # if len(slot2example[l[2:]]) < 2:
                slot2allExample[l[2:]].append(entity)
                i = end
        fw.write(context + '\t' + label + '\n')

slot_labels = []
with open('slot_label.txt', 'r') as fr:
    for line in fr:
        slot_labels.append(line.strip('\n'))
slot_labels_set = set(slot_labels)
# print(slots - slot_labels_set)
# print(slot_labels_set - slots)

with open('labels.txt', 'w') as fw:
    slots_list = [slot for slot in slots]
    slots_list.sort()
    for slot in slots_list:
        if slot == 'O': continue
        slot_split = slot.split('.')
        try:
            desp = slot_split[1] + ' of ' + slot_split[0]
        except:
            desp = slot
        desp = desp.replace("_", " ").replace("loc", " location")
        # desp = ' '.join(re.split('[._]', slot))
        exampleCounter = Counter(slot2allExample[slot])
        for example, cnt in exampleCounter.most_common(2):
            slot2example[slot].append(example)
            print(example + ':\t' + str(cnt))
        fw.write(slot + '\t' + desp + '\t' + '\t'.join(slot2example[slot]) + '\n')