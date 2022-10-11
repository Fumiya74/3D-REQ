import os
import json
from tqdm import tqdm
import numpy as np
from id2bbox import bboxTransform
from wordcloud import WordCloud
from collections import Counter
import copy
from edit_file import edit_objects, edit_relationships
from re_generator import dataset_prepare, duplicate_delection2refer

#import tensorflow_datasets as tfds

use_class = ['wall','pillow','chair','shelf','box','table','picture','plant','cabinet','door']
used_classes = {0:'wall',1:'pillow',2:'chair',3:'shelf',4:'box',5:'table',6:'picture',7:'plant',8:'cabinet',9:'door'}
scan_path = "/home/fumiya/matsu/3RScan/data/3RScan/"
datasets_path = "../../data/datasets/"
question_path = "../../data/question.json"
questions_json = open(question_path,"r")
split_info = "../../data/splits/"
scan_info_path = "../../data/3RScan.json"
scan_json = open(scan_info_path,"r")
scan_file = json.load(scan_json)
scene2id_path = "../../data/id2scene_ref.json"
id2word_path = "../../data/id2word.json"
use_class_path = "../../data/used_classses.json"
stat = "../../data/used_dataset_statistics.txt"
test_mesh_path = "/home/fumiya/matsu/3RScan/splits/test_mesh/"
test_pcd_path = "/home/fumiya/matsu/3RScan/splits/test_pcd/"

kotoba = []

splits = ["train", "val", "test"]
split_dict = {}
for split in splits:

    with open(split_info + split + ".txt") as f:
        lines = f.readlines()

    lines = [line.rstrip('\n') for line in lines]
    split_dict[split] = lines
    #print(split_dict)
    
for sf in tqdm(scan_file):
    for sfs in sf["scans"]:
        type = sf["type"]
        if type == "validation":
            type = "val"
        split_dict[type].append(sfs["reference"])
        
q_file = json.load(questions_json)
tokens = []
word2id, id2word = {}, {}
q_label = []

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
""""
with open(use_class_path,'w') as outfile:
    json.dump(used_classes, outfile, indent=2)
with open(id2word_path,'w') as outfile:
    json.dump(id2word, outfile, indent=2)
"""
#encoder = tfds.features.text.TokenTextEncoder(vocab_list,lowercase=True)
print(word2id)
#単語数は30
file_id = 0
file_id_dict = {}
scenes = []

c = 0
c_train, c_val, c_test = 0,0,0
cuc,euc,fuc = 0,0,0

q_list,r_list,l_list = [],[],[]
s_train, s_val, s_test = [],[],[]

r_dict = {}
f_stati = open(stat,"w")
token_length = 0
none_token_length = 0
none_count = 0
max_length = 0
max_bbox_num = 0
min_bbox_num = 100
for q in tqdm(q_file):###TODO###
    #if q["scene_id"] not in ["7272e184-a01b-20f6-8a46-2583655fdd6d"]:
    semseg_path = scan_path + q["scene_id"] +"/semseg.v2.json"
    #print(semseg_path)
    if os.path.exists(semseg_path):
        c = c+1
        scenes.append(q["scene_id"])
        if q["scene_id"] in split_dict["train"]:
            splited_path = "train/"
            split_id = 0
            c_train = c_train+1
            s_train.append(q["scene_id"])
        elif q["scene_id"] in split_dict["val"]:
            splited_path = "val/"
            split_id = 1
            c_val = c_val+1
            s_val.append(q["scene_id"])
        elif q["scene_id"] in split_dict["test"]:
            splited_path = "test/"
            split_id = 2
            c_test = c_test+1
            s_test.append(q["scene_id"])
        else:
            print(q["scene_id"])

        ###CHANGE###    
        q_label.extend(q["question_label"])
        kotoba.extend(q["re_tokens"])
        cuc = cuc + q["current uncertainty"]
        euc = euc + q["expected uncertainty"]
        for qfuc in q["future uncertainty"]:
            fuc = fuc + qfuc
        r_dict["scene_id"] = q["scene_id"]
        r_dict["refer"] = q["refer"]
        r_list.append(copy.copy(r_dict))
        token_length = token_length + len(q["re_tokens"])
        if max_length < len(q["re_tokens"]):
            max_length = len(q["re_tokens"])
        """
        if q["question_label"] == "None":
            none_token_length = none_token_length + len(q["re_tokens"])
            none_count = none_count + 1
        """
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
        if isinstance(q["ids"],list):
            qids = q["ids"]
        else:
            qids = [q["ids"]]
        #if file_id == 1443:
            #print(q["ids"])
        for s in semseg["segGroups"]:
            if str(s["id"]) in qids: #["21"]
                #if file_id == 1443:
                    #print("DEBUG: ",s["id"])    

                bbox = (bboxTransform(s))
                bbox.append(use_class.index(q["label"]))
                bboxes.append(bbox)
        for bp in bbox[3:6]:
            bp = bp/2
            if bp <= 0:
                print("Length is less than 0:",file_id)

        if bbox[7] >  9 or bbox[7] < 0:
            print("Out of range of class label.:",file_id)
        file_id = file_id + 1
        if len(bboxes) > max_bbox_num:
            max_bbox_num = len(bboxes)
        if len(bboxes) < min_bbox_num:
            min_bbox_num = len(bboxes)
        np.save(bbox_path,np.array(bboxes))
        np.savez(caption_path,encoded_re)
        np.savez(question_path,encoded_q)
s_train = list(set(s_train))
s_val = list(set(s_val))
s_test = list(set(s_test))
print("max:",max_bbox_num,"\nmin:",min_bbox_num)
f_stati.writelines('RE単語数平均:'+ str(token_length/c) + "\n")
f_stati.writelines('最大単語数:'+ str(max_length) + "\n")
#f_stati.writelines('質問がNoneになるときのRE単語数平均:'+ str(none_token_length/none_count) + "\n")
unique_refer_num = duplicate_delection2refer(r_list)
f_stati.writelines('RE内容数（文法の違いを含まない）:'+ str(unique_refer_num) + "\n")
f_stati.writelines('RE数:'+ str(c) + "\n")
f_stati.writelines('train_scenes:'+ str(len(s_train)) + "\n")
f_stati.writelines('val_scenes:'+ str(len(s_val)) + "\n")
f_stati.writelines('test_scenes:'+ str(len(s_test)) + "\n")
f_stati.writelines('train_re:'+ str(c_train) + "\n")
f_stati.writelines('val_re:'+ str(c_val) + "\n")
f_stati.writelines('test_re:'+ str(c_test) + "\n")
word_chain = ' '.join(kotoba)
wc = WordCloud(background_color = "white",max_font_size=40,collocations = False).generate(word_chain)
wc.to_file("../../data/wc.png")
question_rank = Counter(q_label)
print('質問の内訳:',question_rank.most_common())
f_stati.writelines('質問の内訳:'+ str(question_rank.most_common()) + "\n")
print("使用したScan数:",len(list(set(scenes))))
f_stati.writelines("現在の不確定性の平均：" + str(cuc/c) + "\n" + "質問後の不確定性の期待値の平均："+ str(euc/c) + "\n" + "質問後の不確定性の平均：" + str(fuc/len(q_list)) + "\n")
f_stati.close()
with open(scene2id_path,'w') as outfile:
    json.dump(file_id_dict, outfile, indent=2)

"""
for origin in s_test:
    mesh_path = scan_path + origin + "/mesh.refined.v2.obj"
"""