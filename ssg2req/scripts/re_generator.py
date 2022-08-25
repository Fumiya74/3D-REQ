import json
import random
from collections import Counter
import copy
from tqdm import tqdm

def flatten_list(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten_list(el)
        else:
            yield el

def segment_token(l):
    for el in l:
        if isinstance(el, list):
            yield from segment_token(el)
        else:
            yield el.split()

def duplicate_delection(questions):
    unique_questions = list(map(json.loads, set(map(json.dumps, questions))))
    return unique_questions

def sen_gen(file):

    color_list = ["white","black","green","blue","red","brown","yellow","gray","orange","purple","pink","beige","bright","dark","light","silver","gold"]
    shape_list = ["round","flat","L-shaped","semicircular","circular","square","rectangular","sloping","cylindrical","oval","bunk","heart-shaped","u-shaped","octagon"]
    size_list = ["big","small","tall","low","narrow","wide"]
    material_list = ["wooden","plastic","metal","glass","stone","leather","concrete","ceramic","brick","padded","cardboard","marbled","carpet","cork","velvet"]
    texture_list = ["striped","patterned","dotted","colorful","checker","painted","shiny","tiled"]
    #state_list = ["new","old","dirty","clean","open","empty","full","hanging","half open/closed","half "]
    specify_words = ["see","touch","look","watch"]
    all_re = []
    for f in file:
        color = []
        shape = []
        size = []
        material = []
        texture = []
        token = []
        relationship = []
        comparative = []

        for refer in f["refer"][1:]:
            if refer in color_list:
                color.append(refer)
            elif refer in shape_list:
                shape.append(refer)
            elif refer in size_list:
                size.append(refer)
            elif refer in material_list:
                material.append(refer)
            elif refer in texture_list:
                texture.append(refer)
            elif isinstance(refer,list):
                if len(refer) == 2:
                    relationship = refer
                if len(refer) == 1:
                    comparative = refer
            else:
                print("リストに何かしら不足あり")
        if len(color) >= 2:
            color.insert(len(color)-1,"and")
        if len(shape) >= 2:
            shape.insert(len(shape)-1,"and")
        if len(size) >= 2:
            size.insert(len(size)-1,"and")
        if len(material) >= 2:
            material.insert(len(material)-1,"and")
        if len(texture) >= 2:
            texture.insert(len(texture)-1,"and")


        rand_speci_word = random.randrange(len(specify_words)+1)
        if rand_speci_word < len(specify_words):
            token.append(specify_words[rand_speci_word])

        token.append('the')
        first = []
        second = []
        col_rand = random.randrange(2)
        sha_rand = random.randrange(2)
        siz_rand = random.randrange(2)
        mat_rand = random.randrange(2)
        tex_rand = random.randrange(2)
        comparative_in_second = False
        if color != []:
            if col_rand == 1:
                first.append(color)
            else:
                second.append(color)
        if shape != []:
            if sha_rand == 1:
                first.append(shape)
            else:

                second.append(shape)
        if size != []:
            if siz_rand == 1:
                first.append(size)
            else:
                second.append(size)
        if material != []:
            if mat_rand == 1 or len(second) != 0:
                first.append(material)
            else:
                for m in material:
                    if m == "wooden":
                        material[material.index(m)] = "wood"
                    if m == "padded":
                        material[material.index(m)] = "pads"
                    if m == "brick":
                        material[material.index(m)] = "bricks"
                
                material.insert(0,"made of")
                second.append(material)
        if texture != []:
            if tex_rand == 1:
                first.extend(texture)
            else:
                second.append(texture)
        random.shuffle(first)
        first.append(f["label"])
        random.shuffle(second)
        if len(second) >= 2:
            for sec_idx in range(len(second)-1):
                second[sec_idx].append("and")
        relationships_is_first = False
        if relationship != []:
            if relationship[1] == f['label']:
                relationship[1] = "the other"
            else:
                relationship.insert(1,"the")
            if relationship[0] == "right":
                relationship[0] = "on the right of"
            if relationship[0] == "left":
                relationship[0] = "on the left of"
            if relationship[0] == "front":
                relationship[0] = "in front of"

            if len(first) <= len(second)+1:
                first.append(relationship)
                relationships_is_first = True
            else:
                if len(second) != 0:
                    second.append("and")
                second.append(relationship)

        if comparative != []:
            if len(second) != 0 or len(first) >= 3 or relationships_is_first:
                if len(second) != 0:
                    second.append("and")
                second.append(comparative)
                second.append("the other")
            else:
                first.insert(0,[comparative[0].split()[0]])
        token.extend(first)
        if len(second) != 0:
            token.append("it is")
            token.extend(second)
        
        #print(list(flatten_list(segment_token(token))))
        referring_expression = list(flatten_list(segment_token(token)))
        f["referring expression"] = " ".join(referring_expression)
        f["tokens"] = referring_expression
        all_re.append(f)
    
    unique_questions = duplicate_delection(all_re)
    return unique_questions


def ref_gen(questions,tar_n,ref_exp,objects,known,unknown):
    distractor_list = []
    max_uncer = 10
    target = objects[tar_n]
    ref_length = len(ref_exp)
    next_objects = []
    n = 0
    start = False
    for object in objects:
        if object['scene_id'] == target['scene_id']:
            start = True
        if start:
            if object['scene_id'] != target['scene_id']:
                break
            obj_attributes = object['attributes']
            obj_exp = [object['label']]
            obj_exp.extend(obj_attributes.get('color',[]))
            obj_exp.extend(obj_attributes.get('shape',[]))
            obj_exp.extend(obj_attributes.get('size',[]))
            obj_exp.extend(obj_attributes.get('material',[]))
            obj_exp.extend(obj_attributes.get('texture',[]))
            obj_exp.extend(object['relationships'])

            inclusion = True
            for r in ref_exp:
                if r not in obj_exp:
                    inclusion = False
            
            if inclusion:
                distractor_list.append(object['id'])
                next_objects.append(object)
                if object == target:
                    next_tar_n = n
                n = n + 1

    
    uncertainty = len(distractor_list)-1

    if uncertainty <= max_uncer:

        if uncertainty == 0:
            none_restrictor = random.randrange(2)
            if none_restrictor == 1:
                q = {"scene_id":target['scene_id'],"label":target['label'],"refer":ref_exp,"ids":distractor_list,"question":'None'}
                questions.append(q)
        else:
            comparable = False
            if uncertainty == 1:
                for compara in target['comparatives']:
                    if compara[1] in distractor_list:
                        com_exp = [compara[0]]
                        comparable = True
                                    
            if comparable:
                if com_exp[0] in ["bigger than","smaller than","higher than","lower than"]:
                    q = {"scene_id":target['scene_id'],"label":target['label'],"refer":ref_exp,"ids":distractor_list,"question":'size'}
                    questions.append(q) 
                if com_exp[0] in ["darker than","brighter than"]:
                    q = {"scene_id":target['scene_id'],"label":target['label'],"refer":ref_exp,"ids":distractor_list,"question":'color'}
                    questions.append(q)
                none_restrictor = 1 #必ず追加
                if none_restrictor == 1:
                    refer = copy.copy(ref_exp)
                    refer.append(com_exp)                         
                    q = {"scene_id":target['scene_id'],"label":target['label'],"refer":refer,"ids":target['id'],"question":'None'}
                    questions.append(q)
            else:
                random.shuffle(unknown)
                min = uncertainty
                for unk_att in unknown:
                    tmp = []
                    for next_object in next_objects:
                        if unk_att in ['relationships']:
                            if next_object[unk_att] == []:
                                tmp.extend(['None'])
                            else:
                                tmp.extend(next_object[unk_att])
                                tmp = [tuple(i) for i in tmp]
                        else:
                            tmp.extend(next_object['attributes'].get(unk_att,['None']))                   
                    tmp_c = Counter(tmp)
                    tmp_sum = [row[1] for row in tmp_c.most_common()]
                    future_uncertainty = sum(list(map(lambda x:x**2, tmp_sum)))/(len(tmp))-1
                    if future_uncertainty <= min:
                        min = future_uncertainty
                        q_attribute = unk_att
                if min != uncertainty:
                    next_ref_exp = ref_exp
                    refer = copy.copy(ref_exp)
                    if q_attribute in ['relationships']:
                        if objects[tar_n][q_attribute] != []:
                            next_ref_exp.extend([random.choice(objects[tar_n][q_attribute])])
                    else:
                        next_ref_exp.extend(objects[tar_n]['attributes'].get(q_attribute,[]))
                    if len(next_ref_exp) != ref_length:
                        unknown.remove(q_attribute)
                        q = {"scene_id":target['scene_id'],"label":target['label'],"refer":refer,"ids":distractor_list,"question":q_attribute}
                        questions.append(q)
                        if q_attribute == "color":
                            questions.append(copy.copy(q))
                            questions.append(copy.copy(q))
                            questions.append(copy.copy(q))
                        if q_attribute == "size":
                            questions.append(copy.copy(q))
                                

                        if unknown != []:
                            ref_gen(questions,next_tar_n,next_ref_exp,next_objects,known,unknown)

    return questions


def dataset_prepare(objects):

    questions = []

    for tar_object in tqdm(objects):
        tar_label = tar_object['label']
        if all([tar_label != 'floor',tar_label != 'ceiling',tar_object['only'] == False]):
            tar_n = objects.index(tar_object)
            attributes = tar_object['attributes']
            ref_exp = [tar_label]
            known = []
            unknown = []
            color_flag = random.randrange(6)
            shape_flag = random.randrange(6)
            size_flag = random.randrange(6)
            material_flag = random.randrange(6)
            texture_flag = random.randrange(6)
            s_relation_flag = random.randrange(6)      
            if color_flag == 1:
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
          
            ref_gen(questions,tar_n,ref_exp,objects,known,unknown)
    print(len(questions))
    questions = sen_gen(questions)
    print(len(questions))
    return questions
