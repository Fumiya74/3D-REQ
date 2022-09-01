import os
import json
from tqdm import tqdm
import numpy as np
import tensorflow_datasets as tfds

use_class = ['wall','pillow','chair','shelf','box','table','picture','plant','cabinet','door']
scan_path = "/home/fumiya/matsu/3RScan/"
datasets_path = "/home/fumiya/matsu/CyREx/data/datasets/"
question_path = "../../data/question.json"
questions_json = open(question_path,"r")

q_file = json.load(questions_json)
tokens = []
for q in tqdm(q_file):
    tokens.extend(q["ref_tokens"])
    tokens.extend(q["q_tokens"])

vocab_list = list(set(tokens))
encoder = tfds.features.text.TokenTextEncoder(vocab_list,lowercase=True)

#単語数は30
for q in tqdm(q_file):
    semseg_path = scan_path + q["scene_id"] +"/semseg.v2.json"
    caption_path = datasets_path + q["scene_id"] + "_" + q["ref_id"] + "_captions.npz"
    bbox_path = datasets_path + q["scene_id"] + "_" + q["ref_id"] + "_bbox.npy"
    question_path = datasets_path + q["scene_id"] + "_" + q["ref_id"] + "_question.npz"
    semseg_json = open(semseg_path,"r")
    semseg = json.load(semseg_json)
    bboxes = []
    encoded_re = encoder.encode(q["referring expression"])
    encoded_q = encoder.encode(q["question"])
    encoded_re.insert("<start>",0)
    encoded_q.insert("<start>",0)
    

    for s in semseg["segGroups"]:
        if s["id"] in q["ids"]:
            bbox = []
            bbox.extend(s["centroid"])
            bbox.extend(s["axesLength"])
            bbox.append(0)
            bbox.append(use_class.index(q["label"]))
            bboxes.append(bbox)
    
    np.save(bbox_path,bboxes)


    

            

            
          
           




