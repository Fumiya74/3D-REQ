import random
from collections import Counter
import copy

def Ref_Gen(questions,tar_n,ref_exp,objects,unknown):
        #print(ref_exp,unknown)
        #ref_exp_set = set(ref_exp)
        distractor_list = []
        max_uncer = 10
        target = objects[tar_n]
        ref_detail = len(ref_exp)
        #次に渡す辞書リスト
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
                        """"
                        #セット型に直さなくても包含は計算できるっぽい
                        #obj_exp_set = set(obj_exp)
                        #候補オブジェクトを絞る
                        #TypeError: '<=' not supported between instances of 'tuple' and 'str'になっちゃう
                       
                        if ref_exp <= obj_exp:
                                distractor_list.append(object['id'])
                                next_objects.append(object)
                                if object == target:
                                        next_tar_n = n
                                n = n + 1
                        """
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
        #print(uncertainty)
        if uncertainty <= max_uncer:
                #ここを0にすると質問リストからNoneを取り除ける
                if uncertainty == 0:
                        q = {"scene_id":target['scene_id'],"label":target['label'],"Refer":ref_exp,"ids":distractor_list,"question":'None'}
                        questions.append(q)
                else:
                        random.shuffle(unknown)
                        min = uncertainty
                        for unk_att in unknown:

                                tmp = []
                        
                                for next_object in next_objects:
                                        if unk_att in ['relationships']:
                                                #0除算の防止
                                                if next_object[unk_att] == []:
                                                        tmp.extend(['None'])
                                                else:
                                                        tmp.extend(next_object[unk_att])
                                                        tmp = [tuple(i) for i in tmp]
                                        else:
                                                tmp.extend(next_object['attributes'].get(unk_att,['None']))
                                
                                tmp_c = Counter(tmp)
                                #候補物体に含まれる対象の属性の種類が多い順にカウントされて出力される
                                #print(unk_att,':',tmp_c.most_common())
                                tmp_sum = [row[1] for row in tmp_c.most_common()]
                                #不確定性の期待値の計算
                                #print(tmp)
                                future_uncertainty = sum(list(map(lambda x:x**2, tmp_sum)))/(len(tmp))-1
                                #print('質問した場合の不確定性の期待値：',future_uncertainty)
                                #不確定性の期待値が小さくなる質問内容の更新
                                if future_uncertainty <= min:
                                        min = future_uncertainty
                                        q_attribute = unk_att
                        
                        #どんな質問をしても不確定性が減らない場合を除く
                        if min != uncertainty:
                                next_ref_exp = ref_exp
                                refer = copy.copy(ref_exp)
                                if q_attribute in ['relationships']:
                                        if objects[tar_n][q_attribute] != []:
                                                next_ref_exp.extend([random.choice(objects[tar_n][q_attribute])])

                                else:
                                        next_ref_exp.extend(objects[tar_n]['attributes'].get(q_attribute,[]))

                                #質問した属性をtargetが持っていない場合は質問リストに追加しない
                                if len(next_ref_exp) != ref_detail:
                                        unknown.remove(q_attribute)
                                        q = {"scene_id":target['scene_id'],"label":target['label'],"Refer":refer,"ids":distractor_list,"question":q_attribute}
                                        questions.append(q)
                                        #TODO（優先度低）
                                        #辞書リストの絞り込み
        
                                        if unknown != []:
                                                Ref_Gen(questions,next_tar_n,next_ref_exp,next_objects,unknown)

        return questions




                                
