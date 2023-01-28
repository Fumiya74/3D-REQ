from pc_util import *
import json
import numpy as np

import random

def bboxTransform(object_info):

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

  cb0 = [c0,c1,c2,c3,c4,c5,c6,c7]

  cb = np.array(cb0)
  min_x, max_x, min_y, max_y, min_z, max_z = min(cb[:,0]), max(cb[:,0]), min(cb[:,1]), max(cb[:,1]), min(cb[:,2]), max(cb[:,2])
  
  center_x = (max_x+min_x)/2.0
  center_y = (max_y+min_y)/2.0
  center_z = (max_z+min_z)/2.0
  
  len_x = max_x - min_x
  len_y = max_y - min_y
  len_z = max_z - min_z    
    
  return [center_x,center_y,center_z,len_x,len_y,len_z,0]
  



""""
with open('semseg.v2.json', 'r') as f:
  annotation_file = json.load(f)['segGroups']



for i in range(len(annotation_file)):
  object_info = annotation_file[i]

  print(bboxTransform(object_info))
  exit()

"""
