import os
import json
from tqdm import tqdm
import numpy as np
from id2bbox import bboxTransform

#import tensorflow_datasets as tfds

use_class = ['wall','pillow','chair','shelf','box','table','picture','plant','cabinet','door']
scan_path = "/home/fumiya/matsu/3RScan/Scans/"
datasets_path = "../../data/datasets/"
question_path = "../../data/question.json"
questions_json = open(question_path,"r")
split_info = "../../data/splits/"
scan_info_path = "../../data/3RScan.json"
scan_json = open(scan_info_path,"r")
scan_file = json.load(scan_json)
scene2id_path = "../../data/id2scene_ref.json"

splits = ["train", "val", "test"]
split_dict = {}
for split in splits:

    with open(split_info + split + ".txt") as f:
        lines = f.readlines()

    lines = [line.rstrip('\n') for line in lines]
    split_dict[split] = lines
    print(split_dict)
    
for sf in tqdm(scan_file):
    for sfs in sf["scans"]:
        type = sf["type"]
        if type == "validation":
            type = "val"
        split_dict[type].append(sfs["reference"])
        
q_file = json.load(questions_json)
tokens = []
word2id, id2word = {}, {}

for q in tqdm(q_file):
    tokens.extend(q["re_tokens"])
    tokens.extend(q["q_tokens"])
    
tokens.append("<start>")
tokens.append("<end>")

for word in tokens:
    if word not in word2id:
        id = len(word2id) + 1
        word2id[word] = id
        id2word[id] = word

#encoder = tfds.features.text.TokenTextEncoder(vocab_list,lowercase=True)
print(word2id)
#単語数は30
file_id = 0
file_id_dict = {}
for q in tqdm(q_file):###TODO###
    #if q["scene_id"] not in ["7272e184-a01b-20f6-8a46-2583655fdd6d"]:
    semseg_path = scan_path + q["scene_id"] +"/semseg.v2.json"
    if os.path.exists(semseg_path):

        if q["scene_id"] in split_dict["train"]:
            splited_path = "train/"
            split_id = 0
        elif q["scene_id"] in split_dict["val"]:
            splited_path = "val/"
            split_id = 1
        elif q["scene_id"] in split_dict["test"]:
            splited_path = "test/"
            split_id = 2
        else:
            print(q["scene_id"])

        file_id_dict[file_id] = {"scene_id":q["scene_id"],"ref_id":q["ref_id"],"split":split_id}
        caption_path = datasets_path  + splited_path + f'{file_id:06}' + "_captions.npz"
        bbox_path = datasets_path  + splited_path + f'{file_id:06}' + "_bbox.npy"
        question_path = datasets_path  + splited_path + f'{file_id:06}' + "_question.npz"
        semseg_json = open(semseg_path,"r")
        semseg = json.load(semseg_json)
        bboxes = []
        #encoded_re = encoder.encode(q["referring expression"])
        #encoded_q = encoder.encode(q["question"])
        #[単語数,<start>, , , ,<end>,<pad>,,,]
        encoded_q = [len(q["q_tokens"]),word2id["<start>"]]
        encoded_re = [len(q["re_tokens"]),word2id["<start>"]]

        for q_word in q["q_tokens"]:
            encoded_q.append(word2id[q_word])
        for re_word in q["re_tokens"]:
            encoded_re.append(word2id[re_word])
        
        encoded_q.append(word2id["<end>"])
        encoded_re.append(word2id["<end>"])

        encoded_q = np.pad(np.array(encoded_q), [(0,30-len(encoded_q))])
        encoded_re = np.pad(np.array(encoded_re), [(0,30-len(encoded_re))])

        for s in semseg["segGroups"]:
            if str(s["id"]) in q["ids"]:
                #print(s)
                ##########
                bbox = (bboxTransform(s))
                bbox.append(use_class.index(q["label"]))
                bboxes.append(bbox)

        file_id = file_id + 1

        np.save(bbox_path,np.array(bboxes))
        np.savez(caption_path,encoded_re)
        np.savez(question_path,encoded_q)

with open(scene2id_path,'w') as outfile:
    json.dump(file_id_dict, outfile, indent=2)
