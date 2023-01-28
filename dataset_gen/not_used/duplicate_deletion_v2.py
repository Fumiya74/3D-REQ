from copy import copy
import json
import random
from collections import Counter
import  re_gen_v2
from tqdm import tqdm

def main():
    open_file_path = '../../data/question_sentence_data.json'
    out_file_path = '../../data/question_sentence_v2_data.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)
    """
    q_list = []
    #これだと何故か重複あるやつが1つも残らずに消えちゃう（順番はそのまま）
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
    """
    #辞書型リストの重複を削除する（順番がごっちゃになる）
    unique_q_list = list(map(json.loads, set(map(json.dumps, file))))
    print('RE数:',len(unique_q_list))
    question_list = []
    f = open("../../data/referring_expression.txt","w")
    for q in unique_q_list:
        question_list.append(q['question'])
        f.writelines(" ".join(q["tokens"])+"\n")
    question_rank = Counter(question_list)
    f.close()
    print('質問の内訳:',question_rank.most_common())
    with open(out_file_path,'w') as outfile:
        json.dump(unique_q_list, outfile, indent=2)

main()