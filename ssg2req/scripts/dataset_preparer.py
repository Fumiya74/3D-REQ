import json
import random
from collections import Counter

MAX_UNCERTAINTY = 5
open_file_path = '../../data/edited_objects.json'
out_file_path = '../../data/question_data.json'
file_open = open(open_file_path,'r')
file = json.load(file_open)