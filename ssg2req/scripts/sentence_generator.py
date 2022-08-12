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
        color = [""]
        shape = []
        size = []
        material = []
        texture = []

        for refer in f["refer"][1:]:
            if refer in color_list:
                color.append(refer)
            if refer in shape_list:
                shape.append(refer)
            if refer in size_list:
                size.append(refer)
            if refer in material_list:
                material.append(refer)
            if refer in texture_list:
                texture.append(refer)
            if isinstance(refer,list):
                if len(refer) == 2:
                    relationship = refer
                if len(refer) == 1:
                    comparative = refer
            