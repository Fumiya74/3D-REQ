import json
import random

def main():
    open_file_path = '../../data/question_dataset.json'
    out_file_path = '../../data/question_sentence_data.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)
    color_list = ["white","black","green","blue","red","brown","yellow","gray","orange","purple","pink","beige","bright","dark","light","silver","gold"]
    shape_list = ["round","flat","L-shaped","semicircular","circular","square","rectangular","sloping","cylindrical","oval","bunk","heart-shaped","u-shaped","octagon"]
    size_list = ["big","small","tall","low","narrow","wide"]
    material_list = ["wooden","plastic","metal","glass","stone","leather","concrete","ceramic","brick","padded","cardboard","marbled","carpet","cork","velvet"]
    texture_list = ["striped","patterned","dotted","colorful","checker","painted","shiny","tiled"]
    specify_words = ["choose","select","look"]
    
    for f in file:
        color = []
        shape = []
        size = []
        material = []
        texture = []
        token = []

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
        
        rand_speci_word = random.randrange(len(specify_words)+1)
        if rand_speci_word < len(specify_words):
            token.append(specify_words[rand_speci_word])

        token.append('the')

        col_rand = random.randrange(2)
        sha_rand = random.randrange(2)
        siz_rand = random.randrange(2)
        mat_rand = random.randrange(2)
        tex_ramd = random.randrange(2)
        if == 1:
        if all([col_rand == 1,sha_rand == 1, siz_rand == 1,mat_rand == 1,tex_ramd == 1]):
            token.extend(["it","is"])

        token.extend(relationship)
        if == 0:

    
main()
            