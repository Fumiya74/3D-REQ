#最終的に作りたいフォーマットは全質問がリストになっていて、１つの質問ごとに辞書型で詳細を記述
import json

file_path = "../../data/edited_objects.json"
objects_json = open("../../data/objects.json","r")
objects = json.load(objects_json)

#全てのインスタンスを一つのlistにまとめて格納、各要素は1層のdict
all_objects = []

#scansの中のリストを取り出す
#print(isinstance(objects["scans"],list))
#True
#各要素は辞書型になっている
#print(isinstance(objects["scans"][0],dict))
#True
scenes = objects["scans"]
for scene in scenes:
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
        all_objects.append(object)

with open(file_path,'w') as outfile:
    json.dump(all_objects, outfile, indent=2)