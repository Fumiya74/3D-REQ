import os
import json
from tqdm import tqdm
import numpy as np
from id2bbox import bboxTransform
from pc_util import *
import random

scan_path = "/home/fumiya/matsu/3RScan/Scans/"
datasets_path = "../../data/datasets/"
q_path = "../../data/question.json"
output = "../figure/"
id2re = datasets_path + "/id2scene_ref.json"
file_id = 750

bboxes_path = datasets_path + "train/" + f'{file_id:06}' + "_bbox.npy" #CHANGE

bboxes = np.load(bboxes_path,allow_pickle=True)

id2re_json = open(id2re,"r")
id2re_file = json.load(id2re_json)
q_json = open(q_path,"r")
q_file = json.load(q_json)

for q in q_file:
    if q["scene_id"] == id2re_file[str(file_id)]["scene_id"] and q["ref_id"] == id2re_file[str(file_id)]["ref_id"]:
        re = q["referring expression"]
        ques = q["question"]
        print(re,ques)
i = 0
for bbox in bboxes:
    write_lines_as_cylinders(bbox,'test'+str(i), color=[random.randint(0,255),random.randint(0,255),random.randint(0,255),0])
    i = i+1
