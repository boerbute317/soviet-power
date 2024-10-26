# -*- coding: utf-8 -*-
import json

entity_label_type_set = set()
with open("./datas/medical/training.txt", 'r', encoding="utf-8") as reader:
    for line in reader:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for entity in record['entities']:
            entity_label_type_set.add(entity['label_type'])
entity_label_type_list = ['O']
for entity_label in entity_label_type_set:
    entity_label_type_list.extend([f'B-{entity_label}', f'I-{entity_label}'])
print(entity_label_type_list)
with open('./datas/medical/categories.json', 'w', encoding='utf8') as writer:
    json.dump(entity_label_type_list, writer, ensure_ascii=False, indent=2)
