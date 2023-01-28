from pc_util import *
import json
import numpy as np


def my_compute_box_3d(center, size, heading_angle):
  print(heading_angle)
  R = rotz(-1 * heading_angle)
  l, w, h = size
  l = l/2.0
  w = w/2.0
  h = h/2.0
  x_corners = [-l, l, l, -l, -l, l, l, -l]
  y_corners = [w, w, -w, -w, w, w, -w, -w]
  z_corners = [h, h, h, h, -h, -h, -h, -h]
  corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
  corners_3d[0, :] += center[0]
  corners_3d[1, :] += center[1]
  corners_3d[2, :] += center[2]
  return np.transpose(corners_3d)
  
def get_bbox_from_corners(bed):
  pcl = np.random.rand(12, 2,3)  

  ###########################################3
  pcl[0][0] = [bed[0][0],bed[0][1],(bed[0][2])]
  pcl[0][1] = [bed[1][0],bed[1][1],(bed[1][2])]
  
   
  pcl[1][0] = [bed[1][0],bed[1][1],(bed[1][2])]
  pcl[1][1] = [bed[2][0],bed[2][1],(bed[2][2])]

  
  pcl[2][0] = [bed[2][0],bed[2][1],(bed[2][2])]
  pcl[2][1] = [bed[3][0],bed[3][1],(bed[3][2])]
    
  pcl[3][0] = [bed[3][0],bed[3][1],(bed[3][2])]
  pcl[3][1] = [bed[0][0],bed[0][1],(bed[0][2])]
    

  pcl[4][0] = [bed[4][0],bed[4][1],(bed[4][2])]
  pcl[4][1] = [bed[5][0],bed[5][1],(bed[5][2])]    
    
  pcl[5][0] = [bed[5][0],bed[5][1],(bed[5][2])]      
  pcl[5][1] = [bed[6][0],bed[6][1],(bed[6][2])]        
    
  pcl[6][0] = [bed[6][0],bed[6][1],(bed[6][2])]       
  pcl[6][1] = [bed[7][0],bed[7][1],(bed[7][2])]        
    
  pcl[7][0] = [bed[7][0],bed[7][1],(bed[7][2])] 
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

with open('semseg.v2.json', 'r') as f:
  annotation_file = json.load(f)
  
table_bbox = annotation_file['segGroups'][5]['obb']
lengths = table_bbox['axesLengths']
#lengths[2],lengths[1] = lengths[1],lengths[2]
centers = table_bbox['centroid']

#centers[2],centers[1] = centers[1],centers[2]
cb = my_compute_box_3d(centers,lengths,0)
print(cb)
    
curr_bbox = get_bbox_from_corners(cb)###
    
write_lines_as_cylinders(curr_bbox, 'test', color=[255,0,0,0])


