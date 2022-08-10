import json
import copy
from collections import Counter

file_path = "../../data/relationships.json"
relations_json = open(file_path,"r")
relations = json.load(relations_json)
out1 = "../../data/edited_relationships.json"
out2 = "../../data/edited_comparatives.json"
file_path = "../../data/edited_objects.json"
objects_json = open("../../data/objects.json","r")
objects = json.load(objects_json)
all_scenes = []
scan_dict = {}

mode = 2

#データセットに含めるrelationshipsのIDリスト
if mode == 1:
    r_list = [1,2,3,4,5,6,7,14,15,16,17,18,19,23,24,25,26]

if mode == 2:
    r_list = [8,9,10,11,33,34,35,36,37,38,39]


scenes = relations["scans"] #リスト型
for scene in scenes:
    all_relat = []
    scan_dict['scene_id'] = scene['scan']
    #print(scan_dict['scene_id'])
    for relationship in scene['relationships']:
        #含まれる場合追加
        if relationship[2] in r_list:
            all_relat.append(relationship)
    
    scan_dict['relationships'] = all_relat
    #print(scan_dict)
    #copyしないと同じやつがずっと追加されちゃう
    all_scenes.append(scan_dict.copy())

#TODO アンカーをID表記でなくクラス名表記にして重複を削除

all_scenes2 = []
obj_scenes = objects["scans"]
for scene in obj_scenes:
    scene_id = scene["scan"] 
    objects_list = scene["objects"]
    objects_label_list = []
    #床とか天井とかはアンカーとして不向きなので除く
    no_anchor_class = ['floor','ceiling']
    relexist = False
    for object in objects_list:
        objects_label_list.append(object['label']) 
    
    objects_label_c = Counter(objects_label_list)
    for row in objects_label_c.most_common():
        if row[1] >=3:
            no_anchor_class.append(row[0])

    """"
    print(scene_id)
    print(objects_label_c)
    print(no_anchor_class)
    """
    
    scene_dict = {}
    scene_dict['scene_id'] = scene_id

    for relationships in all_scenes:
        if relationships['scene_id'] == scene_id:
            relexist = True
            scene_dict['no_anchor_class'] = no_anchor_class
            relationships_list = []
            for relationship in relationships['relationships']:
                for object2 in objects_list:
                    if str(relationship[1]) == object2['id']:
                        relationship.append(object2['label'])
                        relationships_list.append(relationship)
            scene_dict['relationships'] = relationships_list
    if relexist:
        all_scenes2.append(scene_dict.copy())
if mode == 1:
    with open(out1,'w') as outfile:
        json.dump(all_scenes2, outfile, indent=2)
if mode == 2:
    with open(out2,'w') as outfile:
        json.dump(all_scenes2, outfile, indent=2)

