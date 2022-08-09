"""""

データセットv0（attributeのみ リスト表現） 作成用

"""""
import json
import random
from collections import Counter
import re_gem
from tqdm import tqdm

#最初に渡すobjectsを同一シーンに絞る
def main():
    open_file_path = '../../data/edited_objects.json'
    out_file_path = '../../data/question_data.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)

    questions = []
    #ok = True
    #tar_object = file[990]
    for tar_object in tqdm(file):
        tar_label = tar_object['label']
        if all([tar_label != 'wall',tar_label != 'floor',tar_label != 'ceiling']):
            tar_n = file.index(tar_object)
            #if tar_n >= 990:
                #break
            attributes = tar_object['attributes']
            ref_exp = [tar_label]
            unknown_attributes = []
            color_flag = random.randrange(2)
            shape_flag = random.randrange(2)
            size_flag = random.randrange(2)
            material_flag = random.randrange(2)
            texture_flag = random.randrange(2)
            if color_flag == 1:
                #colorなしの場合は空リスト
                color = attributes.get('color',[])
            else: color = []
            if shape_flag == 1:
                shape = attributes.get('shape',[])
            else: shape = []
            if size_flag == 1:
                size = attributes.get('size',[])
            else: size = []
            if material_flag == 1:
                material = attributes.get('material',[])
            else: material = []
            if texture_flag == 1:
                texture = attributes.get('texture',[])
            else: texture = []

            #Referring Expressionに含まれない
            if color == []:
                unknown_attributes.append('color')
            if shape == []:
                unknown_attributes.append('shape')
            if size == []:
                unknown_attributes.append('size')
            if material == []:
                unknown_attributes.append('material')
            if texture == []:
                unknown_attributes.append('texture')

            ref_exp.extend(color)
            ref_exp.extend(shape)
            ref_exp.extend(size)
            ref_exp.extend(material)
            ref_exp.extend(texture)
            
            re_gem.Ref_Gen(questions,tar_n,ref_exp,file,unknown_attributes)
    #print("a")
    print(questions)
    print(len(questions))
    q_list = []
    l_list = []
    for q in questions:
        q_list.append(q['question'])
        l_list.append(q['label'])
    label_rank = Counter(l_list)
    question_rank = Counter(q_list)
    print(label_rank.most_common())
    print(question_rank.most_common())

main()
