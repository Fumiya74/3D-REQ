from tqdm import tqdm
import copy
from collections import Counter
full = ["empty","full","half full/empty"]
open = ["half open/closed", "open", "closed"]
hanging = ["hanging"]
def edit_objects(scenes):
   
    all_objects = []
    for scene in tqdm(scenes):
        scene_id = scene["scan"] 
        objects_list = scene["objects"]
        for object in objects_list:
            object['scene_id'] = scene_id
            object.pop('ply_color')
            #object.pop('nyu40')
            object.pop('eigen13')
            object.pop('rio27')
            object.pop('affordances',None)
            object['only'] = True
            s_list = []
            full_list = []
            open_list = []
            state = object["attributes"].get('state',[])
            if state != []:
                #この２種類以外のstateが入っていた場合、空リストが格納される
                for s in state:
                    if s in full:
                        full_list.append(s)
                    if s in open:
                        open_list.append(s)
                if full_list != []:
                    object["attributes"]["full"] = full_list
                if open_list != []:
                    object["attributes"]["open"] = open_list
            for other_obj in objects_list:
                if object['label'] == other_obj['label'] and object['id'] != other_obj['id']:
                    object['only'] = False
            #if object['label'] in ['wall','pillow','chair','shelf','box','table','picture','plant','cabinet','door']:#,'kitchen cabinet','window','bag','item','curtain','armchair','towel','light']:
            all_objects.append(object)
    
    return all_objects

def edit_relationships(rel_scenes,objects,r_list):
    #TODO
    all_scenes1 = []
    scans = {}
    for scene1 in tqdm(rel_scenes):
        all_relat = []
        scans['scene_id'] = scene1['scan']

        for relationship in scene1['relationships']:

            if relationship[2] in r_list:
                all_relat.append(relationship)
        
        scans['relationships'] = all_relat
        all_scenes1.append(scans.copy())
    
    all_scenes2 = []
    obj_scenes = objects["scans"]
    for scene in obj_scenes:
        scene_id = scene["scan"] 
        objects_list = scene["objects"]
        objects_label_list = []
        no_anchor_class = ['floor','ceiling']
        relexist = False
        for object in objects_list:
            objects_label_list.append(object['label']) 
        
        objects_label_c = Counter(objects_label_list)
        for row in objects_label_c.most_common():
            #シーン内に3つ以上オブジェクトが含まれているクラスをno_anchor_classに追加
            if row[1] >=3:
                no_anchor_class.append(row[0])

           
        scene_dict = {}
        scene_dict['scene_id'] = scene_id

        for relationships in all_scenes1:
            if relationships['scene_id'] == scene_id:
                relexist = True
                scene_dict['no_anchor_class'] = no_anchor_class
                relationships_list = []
                for relationship in relationships['relationships']:
                    for object2 in objects_list:
                        if str(relationship[1]) == object2['id']:
                            relationship.append(object2['label'])
                            relationship.append(object2["nyu40"])
                            relationships_list.append(relationship)
                scene_dict['relationships'] = relationships_list
        if relexist:
            all_scenes2.append(scene_dict.copy())
        
    return all_scenes2
