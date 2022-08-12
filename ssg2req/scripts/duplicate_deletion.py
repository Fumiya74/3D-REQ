from copy import copy
import json
import random
from collections import Counter
import  re_gen_v2
from tqdm import tqdm

def main():
    open_file_path = '../../data/question_dataset.json'
    out_file_path = '../../data/question_dataset.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)
    q_list = []
    
    for q1 in tqdm(file):
        #リストの抽出表記
        duplicate = False
        for q2 in file[file.index(q1)+1:]:
            if all([q1['scene_id'] == q2['scene_id'], q1['refer'] == q2['refer'], q1['question'] == q2['question']]):
                duplicate = True
                break
        
        if not duplicate:
            q_list.append(q1)

    print(q_list)
    print('RE数:',len(q_list))
    question_list = []
    l_list = []
    for q3 in q_list:
        question_list.append(q3['question'])
        #l_list.append(q['label'])
    label_rank = Counter(l_list)
    question_rank = Counter(question_list)
    #print(label_rank.most_common())
    print('質問の内訳:',question_rank.most_common())
    with open(out_file_path,'w') as outfile:
        json.dump(q_list, outfile, indent=2)

main()