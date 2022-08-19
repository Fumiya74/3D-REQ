from copy import copy
import json
from os import nice
import random
from this import s
#TODO
#1つのAttributeに3つ以上入っちゃっていないか調べる（入っちゃてた）
#narrowとwideがどっちも入っちゃってるやつはなんだ
#smallerとsmallが一緒になっちゃってるやつはあるのか
#on the right(left) of
#in front of

#リストの平易化
def flatten_list(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten_list(el)
        else:
            yield el
#単語ごとに分割
def segment_token(l):
    for el in l:
        if isinstance(el, list):
            yield from segment_token(el)
        else:
            yield el.split()

def main():
    open_file_path = '../../data/question_data.json'
    out_file_path = '../../data/question_sentence_data.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)
    color_list = ["white","black","green","blue","red","brown","yellow","gray","orange","purple","pink","beige","bright","dark","light","silver","gold"]
    shape_list = ["round","flat","L-shaped","semicircular","circular","square","rectangular","sloping","cylindrical","oval","bunk","heart-shaped","u-shaped","octagon"]
    size_list = ["big","small","tall","low","narrow","wide"]
    material_list = ["wooden","plastic","metal","glass","stone","leather","concrete","ceramic","brick","padded","cardboard","marbled","carpet","cork","velvet"]
    texture_list = ["striped","patterned","dotted","colorful","checker","painted","shiny","tiled"]
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

        #1だったらクラス名に修飾、0だったらit is 以下
        first = []
        second = []
        #リストをシャッフルして、その順番で出力するのがいいかも
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
                #TODO
                #material毎にIF文で文法の定義
                #2つ以上の場合もあるのでfor文で回す
                #wood、pads、bricks以外はそのまま
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
        #これ以降はシャッフルしない
        if len(second) >= 2:
            for sec_idx in range(len(second)-1):
                second[sec_idx].append("and")
        relationships_is_first = False
        if relationship != []:
            if relationship[1] == f['label']:
                relationship[1] = "the other"
            else:
                relationship.insert(1,"the")
            #TODO
            #ランダムで言い換えてバリエーションを増やす
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
        
        print(list(flatten_list(segment_token(token))))
        #count = count +1
        #print(count)
        referring_expression = list(flatten_list(segment_token(token)))
        f["referring expression"] = " ".join(referring_expression)
        f["tokens"] = referring_expression
        all_re.append(f)

    with open(out_file_path,'w') as outfile:
        json.dump(all_re, outfile, indent=2)
            


#TODO
# とりあえずTOKEN分けする前の状態で出力させる        
# Noneを減らす前のファイルに対して実行して、偏りをなくす   
main()
            