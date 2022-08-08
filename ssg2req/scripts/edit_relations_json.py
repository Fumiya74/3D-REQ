import json

file_path = "../../data/relationships.json"
relations_json = open(file_path,"r")
relations = json.load(relations_json)
out = "../../data/edited_relationships.json"

all_scenes = []
scan_dict = {}

#データセットに含めるrelationshipsのIDリスト
spatial_list = [1,2,3,4,5,6,7,14,15,16,17,18,19,23,24,25,26]

scenes = relations["scans"]
for scene in scenes:
    all_relat = []
    scan_dict['scene_id'] = scene['scan']
    for relationship in scene['relationships']:
        #含まれる場合追加
        if relationship[2] in spatial_list:
            all_relat.append(relationship)
    
    scan_dict['relationships'] = all_relat
    all_scenes.append(scan_dict)



with open(out,'w') as outfile:
    json.dump(all_scenes, outfile, indent=2)

