"""""

データセットv1（object-object spatial relationships含む リスト表現） 作成用

"""""
import json
import random
from collections import Counter
import  re_gen_v2
from tqdm import tqdm


def main():
    open_file_path = '../../data/edited_objects.json'
    out_file_path = '../../data/question_data.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)

    questions = []

    for tar_object in tqdm(file):
        tar_label = tar_object['label']
        if all([tar_label != 'floor',tar_label != 'ceiling',tar_object['only'] == False]):
            tar_n = file.index(tar_object)
            attributes = tar_object['attributes']
            ref_exp = [tar_label]
            known = []
            unknown = []
            #対象の属性が拾われる確率
            color_flag = random.randrange(6)
            shape_flag = random.randrange(6)
            size_flag = random.randrange(6)
            material_flag = random.randrange(6)
            texture_flag = random.randrange(6)
            #
            s_relation_flag = random.randrange(6)

       
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
            if s_relation_flag == 1:
                if tar_object['relationships'] != []:
                    s_relation = [random.choice(tar_object['relationships'])]
                else:
                    s_relation = []
            else:
                s_relation = []

            #Referring Expressionに含まれない
            if color == []:
                unknown.append('color')
            else:
                known.append('color')
            if shape == []:
                unknown.append('shape')
            else:
                known.append('shape')
            if size == []:
                unknown.append('size')
            else:
                known.append('size')
            if material == []:
                unknown.append('material')
            else:
                known.append('material')
            if texture == []:
                unknown.append('texture')
            else:
                known.append('texture')
            if s_relation == []:
                unknown.append('relationships')
            else:
                known.append('relation')

            ref_exp.extend(color)
            ref_exp.extend(shape)
            ref_exp.extend(size)
            ref_exp.extend(material)
            ref_exp.extend(texture)
            ref_exp.extend(s_relation)

            
            
            re_gen_v2.Ref_Gen(questions,tar_n,ref_exp,file,known,unknown)
    #print("a")
    #print(questions)
    print('RE数:',len(questions))
    q_list = []
    l_list = []
    for q in questions:
        q_list.append(q['question'])
        l_list.append(q['label'])
    label_rank = Counter(l_list)
    question_rank = Counter(q_list)
    #print(label_rank.most_common())
    print('質問の内訳:',question_rank.most_common())
    with open(out_file_path,'w') as outfile:
        json.dump(questions, outfile, indent=2)

main()
