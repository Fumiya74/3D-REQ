import json
from operator import index
import statistics
from tqdm import tqdm
from collections import Counter
import copy
from edit_file import edit_objects, edit_relationships
from re_generator import dataset_prepare, duplicate_delection2refer

relationships_path = "../../data/relationships.json"
objects_path = "../../data/objects.json"

out_path = "../../data/question.json"
q_txt = "../../data/questions.txt"
re_txt = "../../data/referring_expression.txt"
stat = "../../data/dataset_statistics.txt"

objects_json = open(objects_path,"r")
relationships_json = open(relationships_path,"r")

objects_file = json.load(objects_json)
relationships_file = json.load(relationships_json)

relationships_list = [1,2,3,4,5,6,7,14,15,16,17,18,19,23,24,25,26]
bidirectional_list = [6,18]
comparatives_list = [8,9,10,11,33,34,35,36,37,38,39]

def main():
    print("1/5 editting object.json")
    objects = edit_objects(objects_file["scans"])
    print("2/5 editting relationships.json")
    relationships = edit_relationships(relationships_file["scans"],objects_file,relationships_list)
    comparatives = edit_relationships(relationships_file["scans"],objects_file,comparatives_list)
    print("3/5 adding relationships to object.json")
    for object in tqdm(objects):
        object_relationships = []
        object_comparatives = []
        for scene1 in relationships:
            if scene1["scene_id"] == object["scene_id"]:
                for relationship in scene1["relationships"]:
                    if str(relationship[0]) == object['id']:   
                        if relationship[4] not in scene1['no_anchor_class']:
                            if object['label'] != relationship[4] or relationship[2] not in bidirectional_list:

                                object_relationships.append([relationship[3],relationship[4]])

        object_relationships = [tuple(i) for i in object_relationships]
        object_relationships = list(set(object_relationships))
        object_relationships = [list(i) for i in object_relationships]
        objects[objects.index(object)]['relationships'] = object_relationships

        for scene2 in comparatives:
            if scene2["scene_id"] == object["scene_id"]:
                for comparative in scene2["relationships"]:
                    if str(comparative[0]) == object['id']:
                        object_comparatives.append([comparative[3],str(comparative[1])])
        objects[objects.index(object)]['comparatives'] = object_comparatives
    #with open("./obj.json",'w') as outfile:
        #json.dump(objects, outfile, indent=2)
    print("4/5 generationg questions")
    questions = dataset_prepare(objects)
    
    with open(out_path,'w') as outfile:
        json.dump(questions, outfile, indent=2)

    

    q_list = []
    r_list = []
    r_dict = {}
    cuc = 0
    euc = 0
    fuc = 0
    fr = open(re_txt,"w")
    fq = open(q_txt,"w")
    f_stati = open(stat,"w")
    token_length = 0
    none_token_length = 0
    none_count = 0

    for q in questions:
        q_list.append(q['question_label'])
        r_dict["scene_id"] = q["scene_id"]
        r_dict["refer"] = q["refer"]
        r_list.append(copy.copy(r_dict))
        cuc = cuc + q["current uncertainty"]
        euc = euc + q["expected uncertainty"]
        fuc = fuc + q["future uncertainty"]
        fr.writelines(q["referring expression"]+"\n")
        fq.writelines(q["question"]+"\n")
        token_length = token_length + len(q["tokens"])
        if q["question_label"] == "None":
            none_token_length = none_token_length + len(q["tokens"])
            none_count = none_count + 1


    fr.close()
    fq.close()
    f_stati.writelines('RE単語数平均:'+ str(token_length/len(questions)) + "\n")
    f_stati.writelines('質問がNoneになるときのRE単語数平均:'+ str(none_token_length/none_count) + "\n")
    question_rank = Counter(q_list)
    unique_refer_num = duplicate_delection2refer(r_list)
    #print(r_list)
    print('RE数:',len(questions))
    f_stati.writelines('RE内容数（文法の違いを含まない）:'+ str(unique_refer_num) + "\n")
    f_stati.writelines('RE数:'+ str(len(questions)) + "\n")
    #print(label_rank.most_common())
    print('質問の内訳:',question_rank.most_common())
    f_stati.writelines('質問の内訳:'+ str(question_rank.most_common()) + "\n")
    print("現在の不確定性の平均：",cuc/len(questions),"質問後の不確定性の期待値の平均：",euc/len(questions),"質問後の不確定性の平均：",fuc/len(questions))
    f_stati.writelines("現在の不確定性の平均：" + str(cuc/len(questions)) + "\n" + "質問後の不確定性の期待値の平均："+ str(euc/len(questions)) + "\n" + "質問後の不確定性の平均：" + str(fuc/len(questions)) + "\n")
    f_stati.close()

if __name__ == "__main__": main()