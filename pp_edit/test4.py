from pc_util import *
import json
import numpy as np

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

def bboxTransform(cb0):
  cb = np.array(cb0)
  min_x, max_x, min_y, max_y, min_z, max_z = min(cb[:,0]), max(cb[:,0]), min(cb[:,1]), max(cb[:,1]), min(cb[:,2]), max(cb[:,2])
  
  return [[min_x,min_y,min_z],
          [min_x,min_y,max_z],
          [min_x,max_y,min_z],
          [min_x,max_y,max_z],
          [max_x,min_y,min_z],
          [max_x,min_y,max_z],
          [max_x,max_y,min_z],
          [max_x,max_y,max_z]]
###################################################################################





with open('semseg.v2.json', 'r') as f:
  annotation_file = json.load(f)
  
kk = len(annotation_file['segGroups'])  
for i in range(kk):
  object_info = annotation_file['segGroups'][i]

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

  bboxTransform(cb)

  curr_bbox = get_bbox_from_corners(bboxTransform(cb))###
    
  write_lines_as_cylinders(curr_bbox, 'test'+str(i), color=[random.randint(0,255),random.randint(0,255),random.randint(0,255),0])


