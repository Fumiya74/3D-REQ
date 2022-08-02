import random
from collections import Counter
#再帰の時の
#不確定性を計算して次のREを生成する関数 
#引数は（質問リスト、ターゲットのID、参照表現、ディストラクタリスト、不確定要素リスト）
#絞り込んだオブジェクトのみの辞書リストも作成
#出力は不確定性が減らなくなったところまでの質問データセットの辞書リスト
def Ref_Gen(questions,tar_n,ref_exp,objects,unknown_attributes):

        ref_exp_set = set(ref_exp)
        distractor_list = []
        distractor_index_list = []
        max_uncer = 5
        target = objects[tar_n]
        ref_detail = len(ref_exp)
        for object in objects:
                obj_attributes = object['attributes']
                obj_exp = [object['label']]
                obj_exp.extend(obj_attributes.get('color',[]))
                obj_exp.extend(obj_attributes.get('shape',[]))
                obj_exp.extend(obj_attributes.get('size',[]))
                obj_exp.extend(obj_attributes.get('material',[]))
                obj_exp.extend(obj_attributes.get('texture',[]))

                obj_exp_set = set(obj_exp)
                if ref_exp_set <= obj_exp_set:
                        distractor_list.append(object['id'])
                        distractor_index_list.append(objects.index(object))
        
        uncertainty = len(distractor_list) - 1
        if uncertainty <= max_uncer:
                if uncertainty == 0:
                        q = {"scene_id":target['scene_id'],"Refer":ref_exp,"ids":distractor_list,"question":'None'}
                        questions.append(q)
                else:
                        random.shuffle(unknown_attributes)
                        min = uncertainty
                        for unk_att in unknown_attributes:
                                tmp = []
                                #TODO
                                #以下のfor文再検討
                                for dist_id in distractor_index_list:
                                        tmp.extend(objects[dist_id]['attributes'].get(unk_att,['None']))
                                tmp_c = Counter(tmp)
                                tmp_sum = [row[1] for row in tmp_c.most_common()]
                                future_uncertainty = sum(list(map(lambda x:x**2, tmp_sum)))/(len(tmp)) - 1
                                if future_uncertainty <= min:
                                        min = future_uncertainty
                                        q_attribute = unk_att
                        
                        if min != uncertainty:
                                next_ref_exp = ref_exp
                                next_ref_exp.extend(objects[tar_n]['attributes'].get(q_attribute,[]))

                                if len(next_ref_exp) != ref_detail:
                                        Ref_Gen(questions,tar_n,)


                                #TODO
                                #distractor_index_listを変更するコード追記
                                #そもそもindex_listいらない説
                                #tar_nだけ分かればいいのでは


                                
