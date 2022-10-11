import json
import random
from collections import Counter
import copy
from re import A
from tqdm import tqdm
full = ["empty","full","half full/empty"]
open = ["half open/closed", "open", "closed"]
#use_class = ['wall','pillow','chair','shelf','box','table','picture','plant','cabinet','door']# 10 classes
use_class_ids = ["1","2","3","4","5","6","7","8","9","10","11","12","14","16","24","28","33","34","36","39"] # 20 classes (NYU40)
nyu40 = ["cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refridgerator", "shower_curtain", "toilet", "sink", "bathtub", "otherfurniture"]
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
    unique_questions = []
    #unique_questions = list(map(json.loads, set(map(json.dumps, questions))))
    result = []
    print("5/5 deleting dupulication")
    for i in tqdm(questions):
        if [i["scene_id"],i["referring expression"]] not in result:
            result.append([i["scene_id"],i["referring expression"]])
            unique_questions.append(i)

    return unique_questions

def duplicate_delection2refer(refer):
    unique_referring_expressions = []
    unique_referring_expressions = list(map(json.loads, set(map(json.dumps, refer))))
    return len(unique_referring_expressions)

def sen_gen(file):

    color_list = ["white","black","green","blue","red","brown","yellow","gray","orange","purple","pink","beige","bright","dark","light","silver","gold"]
    shape_list = ["round","flat","L-shaped","semicircular","circular","square","rectangular","sloping","cylindrical","oval","bunk","heart-shaped","u-shaped","octagon"]
    size_list = ["big","small","tall","low","narrow","wide"]
    material_list = ["wooden","plastic","metal","glass","stone","leather","concrete","ceramic","brick","padded","cardboard","marbled","carpet","cork","velvet"]
    texture_list = ["striped","patterned","dotted","colorful","checker","painted","shiny","tiled"]
    #state_list = ["new","old","dirty","clean","open","empty","full","hanging","half open/closed","half full/empty","closed"]
    full_list = ["empty","full","half full/empty"]
    open_list = ["closed","open","half open/closed"]
    specify_words = ["see","touch","look","watch"]
    all_re = []
    current_scene = "none"
    for f in file:
        color = []
        shape = []
        size = []
        material = []
        texture = []
        full = []
        open = []
        token = []
        relationship = f["relationship"]
        comparative = f["comparative"]

        for refer in f["refer"]:
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
            elif refer in full_list:
                full.append(refer)
            elif refer in open_list:
                open.append(refer)
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
        sta_rand = random.randrange(2)

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
        if texture != []:
            if tex_rand == 1:
                first.extend(texture)
            else:
                second.append(texture)
        if open != []:
            if sta_rand == 1:
                first.extend(open)
            else:
                second.append(open)
        if full != []:
            if sta_rand == 1:
                first.extend(full)
            else:
                second.append(full)
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
        random.shuffle(first)
        first.append(f["label"])
        random.shuffle(second)
        if len(second) >= 2:
            for sec_idx in range(len(second)-1):
                second[sec_idx].append("and")
        relationships_is_first = False
        if relationship != []:
            expression_selector = random.randrange(2)
            if relationship[1] in f['label']:
                relationship[1] = "the other"
            else:
                relationship.insert(1,"the")
            if relationship[0] == "right":
                relationship[0] = "on the right of"
            if relationship[0] == "left":
                relationship[0] = "on the left of"
            if relationship[0] == "front":
                relationship[0] = "in front of"
            if relationship[0] == "close by" and expression_selector == 1:
                relationship[0] = "near"
            

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
                if comparative[0].split()[0] == "more":
                    first.insert(0,comparative[0])
                else:
                    first.insert(0,[comparative[0].split()[0]])
        token.extend(first)
        token.append(".")
        if len(second) != 0:
            token.append("it is")
            token.extend(second)
            token.append(".")
        
        #print(list(flatten_list(segment_token(token))))
        referring_expression = list(flatten_list(segment_token(token)))
        f["referring expression"] = " ".join(referring_expression)
        f["re_tokens"] = referring_expression
        q_t = []
        for q in f["question"]:
            q_t.append(q.split())
        f["q_tokens"] = q_t
        if f["scene_id"] != current_scene:
            ref_id = 0
            current_scene = f["scene_id"]
        f["ref_id"] = ref_id
        ref_id = ref_id + 1
        all_re.append(f)
    
    unique_questions = duplicate_delection(all_re)
    return unique_questions


def ref_gen(questions,tar_n,ref_exp,s_relat,objects,unknown):
    distractor_list = []
    max_uncer = 10
    target = objects[tar_n]
    tar_relat = copy.copy(s_relat)
    nyu_id = target["nyu40"]
    ref_length = len(ref_exp)+len(s_relat)
    next_ref_length =0
    next_objects = []
    unknown2 = []
    n = 0
    start = False
    #print("target",target)
    #print("s_relat",s_relat)
    #print("ref_exp1",ref_exp)
    for object in objects:
        if object['scene_id'] == target['scene_id']:
            start = True
        if start:
            if object['scene_id'] != target['scene_id']:
                break
            obj_attributes = object['attributes']
            #obj_exp = [object['label']]
            obj_exp = []
            obj_exp.extend(obj_attributes.get('color',[]))
            obj_exp.extend(obj_attributes.get('shape',[]))
            obj_exp.extend(obj_attributes.get('size',[]))
            obj_exp.extend(obj_attributes.get('material',[]))
            obj_exp.extend(obj_attributes.get('texture',[]))
            obj_exp.extend(obj_attributes.get('open',[]))
            obj_exp.extend(obj_attributes.get('full',[]))

            #obj_exp.extend(object['relationships'])

            inclusion = False
            
            if s_relat != []:
                #print("DEBUG1")
                for orel in object["relationships"]:                    
                    if orel[0] == s_relat[0] and s_relat[1] in orel[1]:
                        inclusion = True
                        #print("relationshipTRUE")
            else:
                inclusion = True
            if target["nyu40"] != object["nyu40"]:
                inclusion = False
            if target["label"] != object["label"] and target["label"] not in nyu40:
                inclusion = False
                #print("labelFalse")
            for r in ref_exp:
                if r not in obj_exp:
                    inclusion = False
                    #print("refexpFALSE")
            #print(target["label"],object["label"],inclusion)
            if inclusion:
                distractor_list.append(object['id'])
                next_objects.append(object)
                if object == target:
                    next_tar_n = n
                n = n + 1

    
    uncertainty = len(distractor_list)-1
    #print("DEBUG",uncertainty)
    if uncertainty <= max_uncer:

        if uncertainty == 0:
            none_restrictor = random.randrange(2)
            if none_restrictor == 1:
                q = {"target_id":target["id"],"scene_id":target['scene_id'],"label":target['label'],"nyu40":nyu_id ,\
                "refer":ref_exp,"relationship":s_relat,"comparative":[],"ids":distractor_list,"current uncertainty":uncertainty,"expected uncertainty":0,"future uncertainty":[0],\
                "question_label":['None'],"question":["no questions ."]}
                questions.append(q)
        else:
            comparable = False
            com_exp = []
            if uncertainty == 1:
                for compara in target['comparatives']:
                    if compara[1] in distractor_list:
                        com_exp.append([compara[0]])
                        comparable = True

            ###TODO###
            #クエスチョンラベルとかの変数（リスト）
            q_list = []   
            qs_list = []   #questionsentencelist  
            next_uncertainty_list = []                 
            if comparable: 
                q_list.append("comparative")
                next_uncertainty_list.append(0)
                """              
                q = {"target_id":target["id"],"scene_id":target['scene_id'],"label":target['label'],"nyu40":nyu_id ,\
                    "refer":ref_exp,"relationship":s_relat,"ids":distractor_list,"current uncertainty":uncertainty,"expected uncertainty":0,"future uncertainty":0,\
                    "question_label":['comparative']}
                """
                for com in com_exp:
                    ###TO CHANGE###
                    if com[0] == "bigger than":
                        qs_list.append("is it the bigger one ?")
                        qs_list.append("is it the smaller one ?")
                    elif com[0] == "smaller than":
                        qs_list.append("is it the smaller one ?")
                        qs_list.append("is it the bigger one ?")
                    elif com[0] == "higher than":
                        qs_list.append("is it the higher one ?")
                        qs_list.append("is it the lower one ?")
                    elif com[0] == "lower than":
                        qs_list.append("is it the lower one ?")
                    elif com[0] == "messier than":
                        qs_list.append("is it the messier one ?")
                        qs_list.append("is it the cleaner one ?")
                    elif com[0] == "cleaner than":
                        qs_list.append("is it the cleaner one ?")
                        qs_list.append("is it the messier one ?")
                    elif com[0] == "fuller than":
                        qs_list.append("is it the fuller one ?")
                    elif com[0] == "more closed":
                        qs_list.append("is it the more closed one ?")
                        qs_list.append("is it the more open one ?")
                    elif com[0] == "more open":
                        qs_list.append("is it the more open one ?")
                        qs_list.append("is it the more closed one ?")
                    elif com[0] == "brighter than":
                        qs_list.append("is it the brighter one ?")
                        qs_list.append("is it the darker one ?")
                    elif com[0] == "darker than":
                        qs_list.append("is it the darker one ?")
                        qs_list.append("is it the brighter one ?")
                    else:
                        print(com_exp[0],"There is an error in the source code")

                    #questions.append(q)
                    #questions.append(copy.copy(q))
                    #questions.append(copy.copy(q))
                    #questions.append(copy.copy(q))

                    
                #none_restrictor = 1 #必ず追加
                #if none_restrictor == 1:
                refer = copy.copy(ref_exp)
                #refer.append(com_exp[0])          
                random.shuffle(com_exp)
                q = {"target_id":target["id"],"scene_id":target['scene_id'],"label":target['label'],"nyu40":nyu_id ,\
                    "refer":refer,"relationship":s_relat,"comparative":com_exp[0],"ids":target['id'],"current uncertainty":uncertainty,"expected uncertainty":0,"future uncertainty":[0],\
                    "question_label":['None'],"question":["no questions ."]}

                questions.append(q)
            #else:
            random.shuffle(unknown)
            min = uncertainty
            q_attribute = []
            #print(next_objects)
            for unk_att in unknown:
                tmp = []
                #print(len(next_objects))
                #print(unk_att)
                for next_object in next_objects:
                    if unk_att in ['relationships']:
                        if next_object[unk_att] == []:
                            tmp.append(('N', 'o', 'n', 'e'))
                        else:
                            tmp.extend(next_object[unk_att])
                            tmp = [tuple(i) for i in tmp]
                    else:
                        tmp.extend(next_object['attributes'].get(unk_att,['None']))                   
                tmp_c = Counter(tmp)
                tmp_sum = [row[1] for row in tmp_c.most_common()]
                #print(tmp)
                future_uncertainty = sum(list(map(lambda x:x**2, tmp_sum)))/(len(tmp)) - 1

                ###CHANGE###
                
                if future_uncertainty <= min:
                    if future_uncertainty != min:
                        min = future_uncertainty
                        q_attribute = []
                    q_attribute.append(unk_att)

            if min == uncertainty:
                refer = copy.copy(ref_exp)
                q = {"target_id":target["id"],"scene_id":target['scene_id'],"label":target['label'],"nyu40":nyu_id ,\
                    "refer":refer,"relationship":s_relat,"comparative":[],"ids":distractor_list,"current uncertainty":uncertainty,"expected uncertainty":uncertainty,"future uncertainty":[uncertainty],\
                    "question_label":["which"],"question":["which one ?"]}
                questions.append(q)

            if min != uncertainty:
                
                refer = copy.copy(ref_exp)
                ###TODO###
                first = True
                #print("q_attribute",q_attribute)
                for q_att in q_attribute:
                    next_ref_exp = copy.copy(ref_exp)
                    if q_att in ['relationships']:
                        if objects[tar_n][q_att] != []:
                            tar_relat = random.choice(objects[tar_n][q_att])
                            next_ref_length = ref_length + len(tar_relat)
                            #next_ref_exp.extend([tar_relat])
                    else:                        
                        next_ref_exp.extend(objects[tar_n]['attributes'].get(q_att,[]))
                        next_ref_length = len(next_ref_exp) + len(s_relat)
                        #print("q_att",q_att)
                        #print("1",refer,ref_length)
                        #print("2",next_ref_exp,next_ref_length)

                    unknown2 = copy.copy(unknown)
                    unknown2.remove(q_att)
                    next_uncertainty = 0
                        
                    if next_ref_length != ref_length:            
                        for nexobj in next_objects:
                            #print(nexobj)
                            if "relationships" in q_att:
                                if tar_relat in nexobj[q_att]:
                                    next_uncertainty = next_uncertainty + 1
                            elif target["attributes"][q_att] == nexobj["attributes"].get(q_att,[None]):
                                next_uncertainty = next_uncertainty + 1
                    else:
                        next_uncertainty = uncertainty
                    if next_uncertainty == 0:
                        #print(tar_relat)
                        next_uncertainty = uncertainty
                    next_uncertainty_list.append(next_uncertainty-1)
                    #q = {"target_id":target["id"],"scene_id":target['scene_id'],"label":target['label'],"nyu40":nyu_id ,\
                        #"refer":refer,"relationship":s_relat,"ids":distractor_list,"current uncertainty":uncertainty,"expected uncertainty":min,"future uncertainty":next_uncertainty-1,\
                        #"question_label":q_attribute}
                    """
                    for q_att in q_attribute:
                        if ~in~:
                            qquestion.append("what is the color ?")
                    """
                    ###TO CHANGE###
                    q_list.append(q_att)
                    #print("q_list",q_list)
                    if q_att == "color":
                        qs_list.append("what is the color ?")
                    elif q_att == "size":
                        qs_list.append("what is the size ?")
                    elif q_att == "shape":
                        qs_list.append("what is the shape ?")
                    elif q_att == "texture":
                        qs_list.append("what is the texture ?")
                    elif q_att == "material":
                        qs_list.append("what is it made of ?")
                    elif q_att == "full":
                    
                        qs_list.append("is it full ?")
                        qs_list.append("is it empty ?")
                    elif q_att == "open":
                        qs_list.append("is it open ?")
                        qs_list.append("is it closed ?")
                    elif q_att == "relationships":
                        qs_list.append("where is it ?")
                    else:
                        print(q_att,"There is an error in the source code") 
                    #questionsに追加するのは最初だけ 

                    if first:  
                        q = {"target_id":target["id"],"scene_id":target['scene_id'],"label":target['label'],"nyu40":nyu_id ,\
                            "refer":refer,"relationship":s_relat,"comparative":[],"ids":distractor_list,"current uncertainty":uncertainty,"expected uncertainty":min,"future uncertainty":next_uncertainty_list,\
                            "question_label":q_list,"question":qs_list}
                        questions.append(q)
                        first = False
                    """
                    if q_attribute == "color":
                        questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                    if q_attribute == "size":
                        #questions.append(copy.copy(q))
                        questions.append(copy.copy(q))
                    if q_attribute == "shape":
                        questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                    if q_attribute == "texture":
                        questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                    if q_attribute == "material":
                        questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                        #questions.append(copy.copy(q))
                    """       
                    ### for文でq_attributeを回して次のREを生成する。comparativeのときは生成しない。ここをらんだむにする
                    if unknown2 != [] or comparable == False:
                        if next_ref_length != ref_length:
                            ref_gen(questions,next_tar_n,next_ref_exp,tar_relat,next_objects,unknown2)

    return questions


def dataset_prepare(objects):

    questions = []

    for tar_object in tqdm(objects):
        if all([tar_object['nyu40'] not in ["1","2"], tar_object['nyu40'] in use_class_ids,tar_object['only'] == False]):
            tar_n = objects.index(tar_object)
            attributes = tar_object['attributes']
            #ref_exp = [tar_label]
            ref_exp = []
            #known = []
            unknown = []
            color_flag = random.randrange(6)
            shape_flag = random.randrange(6)
            size_flag = random.randrange(6)
            material_flag = random.randrange(6)
            texture_flag = random.randrange(6)
            state_flag = random.randrange(6)
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
            if state_flag == 1:
                open = attributes.get('open',[])
                full = attributes.get('full',[])
            else: 
                open = []
                full = []
            if s_relation_flag == 1:
                if tar_object['relationships'] != []:
                    s_relation = random.choice(tar_object['relationships'])
                else:
                    s_relation = []
            else:
                s_relation = []
            if color == []:
                unknown.append('color')

            if shape == []:
                unknown.append('shape')

            if size == []:
                unknown.append('size')

            if material == []:
                unknown.append('material')

            if texture == []:
                unknown.append('texture')
            
            if open == []:
                unknown.append('open')
            
            if full == []:
                unknown.append('full')

            if s_relation == []:
                unknown.append('relationships')


            ref_exp.extend(color)
            ref_exp.extend(shape)
            ref_exp.extend(size)
            ref_exp.extend(material)
            ref_exp.extend(texture)
            ref_exp.extend(open)
            ref_exp.extend(full)
            #ref_exp.extend(s_relation)
          
            ref_gen(questions,tar_n,ref_exp,s_relation,objects,unknown)
    print(len(questions))
    questions = sen_gen(questions)
    print(len(questions))
    return questions

