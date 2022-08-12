import json
import random

def main():
    open_file_path = '../../data/question_dataset.json'
    out_file_path = '../../data/question_sentence_data.json'
    file_open = open(open_file_path,'r')
    file = json.load(file_open)
    color_list = []
    shape_list = []
    size_list = []
    material_list = []
    texture_list = []
    specify_words = ["choose","select","look"]

    for f in file:
        color = ["white","black","green","blue","red","brown","yellow","gray","orange","purple","pink","beige","bright","dark","light","silver","gold"]
        shape = ["round","flat","L-shaped","semicircular","circular","square","rectangular","sloping","cylindrical","oval","bunk","heart-shaped","u-shaped","octagon"]
        size = ["big","tall","low","narrow","wide"]
        material = ["wooden","plastic","metal","glass","stone","leather","concrete","ceramic","brick","padded","cardboard","marbled","carpet","cork","velvet"]
        texture = ["striped","patterned","dotted","colorful","checker","painted","shiny",""]

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
            