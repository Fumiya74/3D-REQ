#最終的に作りたいフォーマットは全質問がリストになっていて、１つの質問ごとに辞書型で詳細を記述
import json
from tqdm import tqdm

file_path = "../../data/edited_objects.json"
objects_json = open("../../data/objects.json","r")
objects = json.load(objects_json)
r_open_file_path = '../../data/edited_relationships.json'
r_file_open = open(r_open_file_path,'r')
r_file = json.load(r_file_open)
c_open_file_path = '../../data/edited_comparatives.json'
c_file_open = open(c_open_file_path,'r')
c_file = json.load(c_file_open)

#全てのインスタンスを一つのlistにまとめて格納、各要素は1層のdict
all_objects = []

#scansの中のリストを取り出す
#print(isinstance(objects["scans"],list))
#True
#各要素は辞書型になっている
#print(isinstance(objects["scans"][0],dict))
#True
scenes = objects["scans"]

for scene in tqdm(scenes):
    #現在探索中のscan名を変数で記憶
    scene_id = scene["scan"] 
    objects_list = scene["objects"]
    #print(isinstance(objects_list,list))
    #True
    for object in objects_list:
        object['scene_id'] = scene_id
        object.pop('ply_color')
        object.pop('nyu40')
        object.pop('eigen13')
        object.pop('rio27')
        object.pop('affordances',None)
        obj_rel = []
        obj_com = []
        object['only'] = True
        for oth_obj in objects_list:
            if object['label'] == oth_obj['label'] and object['id'] != oth_obj['id']:
                object['only'] = False

        for relationships in r_file:
            if relationships['scene_id'] == scene_id:
                for relationship in relationships['relationships']:
                    #print([str(relationship[0])])
                    if str(relationship[0]) == object['id']:
                        
                        #no_anchor_class以外
                        
                        if relationship[4] not in relationships['no_anchor_class']:
                            #targetとanchorが同じだった場合は双方向relationshipsは削除
                            if object['label'] != relationship[4] or relationship[2] not in [1,6,18,19]:
                                #print([str(relationship[1]),relationship[3]])
                                obj_rel.append([relationship[3],relationship[4]])
                
        #print(obj_rel)
        #2次元リストはセットにわたせなかったから2次元目の要素をタプルに一回変更
        obj_rel = [tuple(i) for i in obj_rel]
        obj_rel = list(set(obj_rel))
        obj_rel = [list(i) for i in obj_rel]
        object['relationships'] = obj_rel
        for comparatives in c_file:
            if comparatives['scene_id'] == scene_id:
                for comparative in comparatives['relationships']:
                    if str(comparative[0]) == object['id']:
                        obj_com.append([comparative[3],str(comparative[1])])
        object['comparatives'] = obj_com
        all_objects.append(object)

with open(file_path,'w') as outfile:
    json.dump(all_objects, outfile, indent=2)