import json
from operator import index
from tqdm import tqdm
from collections import Counter
from edit_file import edit_objects, edit_relationships
from re_generator import dataset_prepare

relationships_path = "../../data/relationships.json"
objects_path = "../../data/objects.json"

out_path = "../../question.json"

objects_json = open(objects_path,"r")
relationships_json = open(relationships_path,"r")

objects_file = json.load(objects_json)
relationships_file = json.load(relationships_json)

relationships_list = [1,2,3,4,5,6,7,14,15,16,17,18,19,23,24,25,26]
bidirectional_list = [1,6,18,19]
comparatives_list = [8,9,10,11,38,39]

def main():
    print("1/ editting object.json")
    objects = edit_objects(objects_file["scans"])
    print("2/ editting relationships.json")
    relationships = edit_relationships(relationships_file["scans"],objects_file,relationships_list)
    comparatives = edit_relationships(relationships_file["scans"],objects_file,comparatives_list)
    print("3/ adding relationships to object.json")
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
    with open("./obj.json",'w') as outfile:
        json.dump(objects, outfile, indent=2)
    
    questions = dataset_prepare(objects)
    
    with open(out_path,'w') as outfile:
        json.dump(questions, outfile, indent=2)

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
    

if __name__ == "__main__": main()