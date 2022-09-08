from pickle import FALSE, TRUE
from pc_util import *
import json
import numpy as np
import sys

import random
  
def get_bbox_from_corners(bed):
  pcl = np.random.rand(12, 2,3)  

  ###########################################3
  
  pcl[0][0] = [bed[0][0],bed[0][1],(bed[0][2])]
  pcl[0][1] = [bed[1][0],bed[1][1],(bed[1][2])]
  
   
  pcl[1][0] = [bed[1][0],bed[1][1],(bed[1][2])]
  pcl[1][1] = [bed[3][0],bed[3][1],(bed[3][2])]

  
  pcl[2][0] = [bed[2][0],bed[2][1],(bed[2][2])]
  pcl[2][1] = [bed[3][0],bed[3][1],(bed[3][2])]
    
  pcl[3][0] = [bed[2][0],bed[2][1],(bed[2][2])]
  pcl[3][1] = [bed[0][0],bed[0][1],(bed[0][2])]
    


  pcl[4][0] = [bed[4][0],bed[4][1],(bed[4][2])]
  pcl[4][1] = [bed[5][0],bed[5][1],(bed[5][2])]    
    
  pcl[5][0] = [bed[5][0],bed[5][1],(bed[5][2])]      
  pcl[5][1] = [bed[7][0],bed[7][1],(bed[7][2])]        
    
  pcl[6][0] = [bed[6][0],bed[6][1],(bed[6][2])]       
  pcl[6][1] = [bed[7][0],bed[7][1],(bed[7][2])]        
    
  pcl[7][0] = [bed[6][0],bed[6][1],(bed[6][2])] 
  pcl[7][1] = [bed[4][0],bed[4][1],(bed[4][2])]



  pcl[8][0] = [bed[0][0],bed[0][1],(bed[0][2])]  
  pcl[8][1] = [bed[4][0],bed[4][1],(bed[4][2])]
            
  pcl[9][0] = [bed[2][0],bed[2][1],(bed[2][2])]    
  pcl[9][1] = [bed[6][0],bed[6][1],(bed[6][2])] 
    
  pcl[10][0] = [bed[1][0],bed[1][1],(bed[1][2])]    
  pcl[10][1] = [bed[5][0],bed[5][1],(bed[5][2])]  
    
  pcl[11][0] = [bed[3][0],bed[3][1],(bed[3][2])]    
  pcl[11][1] = [bed[7][0],bed[7][1],(bed[7][2])]     
  

  return pcl
  
###################################################################################
q_path = "../../data/question.json"
q_json = open(q_path,"r")
q_file = json.load(q_json)
args = sys.argv
q_id = int(args[1])

q = q_file[q_id]
exist = FALSE
with open('/home/fumiya/matsu/3RScan/data/3RScan/' + q["scene_id"] + '/semseg.v2.json', 'r') as f:
  annotation_file = json.load(f)
  
kk = len(annotation_file['segGroups'])  
for i in range(kk):

  object_info = annotation_file['segGroups'][i]
  if str(object_info["id"]) in q["ids"]:
    axes = object_info['obb']['normalizedAxes']
    sizes = object_info['obb']['axesLengths']
    center = object_info['obb']['centroid']

    v0 = np.array(axes[:3])
    v1 = np.array(axes[3:6])
    v2 = np.array(axes[6:])


    s0, s1, s2 = sizes


    center = np.asarray(center)

    c0 = center - s0*v0/2.0 - s1*v1/2.0 - s2*v2/2.0
    c1 = center + s0*v0/2.0 - s1*v1/2.0 - s2*v2/2.0
    c2 = center - s0*v0/2.0 - s1*v1/2.0 + s2*v2/2.0
    c3 = center + s0*v0/2.0 - s1*v1/2.0 + s2*v2/2.0


    c4 = center - s0*v0/2.0 + s1*v1/2.0 - s2*v2/2.0
    c5 = center + s0*v0/2.0 + s1*v1/2.0 - s2*v2/2.0
    c6 = center - s0*v0/2.0 + s1*v1/2.0 + s2*v2/2.0
    c7 = center + s0*v0/2.0 + s1*v1/2.0 + s2*v2/2.0

    cb = [c0,c1,c2,c3,c4,c5,c6,c7]

      
    curr_bbox = get_bbox_from_corners(cb)###
    exist = TRUE
    dir = '/home/fumiya/matsu/3RScan/data/3RScan/' + q_file[q_id]["scene_id"] + "/" + str(q["ref_id"]) + '/'
    if not os.path.exists(dir):  
      os.makedirs(dir) 
    write_lines_as_cylinders(curr_bbox, dir + 'test'+str(i), color=[0,255,255,0])

if exist:
  print(q_file[q_id]["scene_id"]) 
  print(q["ref_id"])
  print(q["referring expression"])
  print(q["question"])
