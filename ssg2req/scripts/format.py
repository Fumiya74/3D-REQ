import os
import json
from tqdm import tqdm
import numpy as np
#import tensorflow_datasets as tfds

use_class = ['wall','pillow','chair','shelf','box','table','picture','plant','cabinet','door']
scan_path = "/home/fumiya/matsu/3RScan/"
datasets_path = "../../data/datasets/"
question_path = "../../data/question.json"
questions_json = open(question_path,"r")

q_file = json.load(questions_json)
tokens = []
word2id, id2word = {}, {}

for q in tqdm(q_file):
    tokens.extend(q["re_tokens"])
    tokens.extend(q["q_tokens"])
    

vocab_list = list(set(tokens))
vocab_list.append("<start>")
vocab_list.append("<end>")
for word in tokens:
    id = len(word2id) + 1
    word2id[word] = id
    id2word[id] = word
#encoder = tfds.features.text.TokenTextEncoder(vocab_list,lowercase=True)

#単語数は30
for q in tqdm(q_file):
    semseg_path = scan_path + q["scene_id"] +"/semseg.v2.json"
    caption_path = datasets_path + q["scene_id"] + "_" + str(q["ref_id"]) + "_captions.npz"
    bbox_path = datasets_path + q["scene_id"] + "_" + str(q["ref_id"]) + "_bbox.npy"
    question_path = datasets_path + q["scene_id"] + "_" + str(q["ref_id"]) + "_question.npz"
    semseg_json = open(semseg_path,"r")
    semseg = json.load(semseg_json)
    bboxes = []
    #encoded_re = encoder.encode(q["referring expression"])
    #encoded_q = encoder.encode(q["question"])
    #[単語数,<start>, , , ,<end>,<pad>,,,]
    encoded_q = [len(q["q_tokens"]),"<start>"]
    encoded_re = [len(q["re_tokens"]),"<start>"]

    for q_word in q["q_tokens"]:
        encoded_q.append(word2id[q_word])
    for re_word in q["re_tokens"]:
        encoded_re.append(word2id[re_word])
    
    encoded_q.append("<end>")
    encoded_re.append("<end>")

    encoded_q = np.pad(np.array(encoded_q), [(0,30-len(encoded_q))])
    encoded_re = np.pad(np.array(encoded_re), [(0,30-len(encoded_re))])

    for s in semseg["segGroups"]:
        if s["id"] in q["ids"]:
            bbox = []
            bbox.extend(s["centroid"])
            bbox.extend(s["axesLength"])
            bbox.append(0)
            bbox.append(use_class.index(q["label"]))
            bboxes.append(bbox)

    np.save(bbox_path,np.array(bboxes))
    np.savez(caption_path,encoded_re)
    np.savez(question_path,encoded_q)
