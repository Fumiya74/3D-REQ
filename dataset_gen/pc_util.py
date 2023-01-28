# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import open3d as o3d

# Point cloud IO
import numpy as np
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

import math
# Mesh IO
import trimesh
#import trimesh.io

import matplotlib.pyplot as pyplot

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)

def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b,:,:], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)

def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize,vsize,vsize,num_sample,3))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i,j,k) not in loc2pc:
                    vol[i,j,k,:,:] = np.zeros((num_sample,3))
                else:
                    pc = loc2pc[(i,j,k)] # a list of (3,) arrays
                    pc = np.vstack(pc) # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0]>num_sample:
                        pc = random_sampling(pc, num_sample, False)
                    elif pc.shape[0]<num_sample:
                        pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i,j,k])+0.5)*voxel - radius
                    pc = (pc - pc_center) / voxel # shift and scale
                    vol[i,j,k,:,:] = pc 
    return vol

def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b,:,:], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2*radius/float(imgsize)
    locations = (points[:,0:2] + radius)/pixel # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i,j) not in loc2pc:
                img[i,j,:,:] = np.zeros((num_sample,3))
            else:
                pc = loc2pc[(i,j)]
                pc = np.vstack(pc)
                if pc.shape[0]>num_sample:
                    pc = random_sampling(pc, num_sample, False)
                elif pc.shape[0]<num_sample:
                    pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                pc_center = (np.array([i,j])+0.5)*pixel - radius
                pc[:,0:2] = (pc[:,0:2] - pc_center)/pixel
                img[i,j,:,:] = pc
    return img
# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def read_ply_rgb(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z,r,g,b in pc])
    return pc_array

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    
    vertex = []
    #colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]    
    colors = [colormap(i/float(num_classes)) for i in range(num_classes)]    
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
   
def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()

# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)

# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)     
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points-ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


# ----------------------------------------
# BBox
# ----------------------------------------
def bbox_corner_dist_measure(crnr1, crnr2):
    """ compute distance between box corners to replace iou
    Args:
        crnr1, crnr2: Nx3 points of box corners in camera axis (y points down)
        output is a scalar between 0 and 1        
    """
    
    dist = sys.maxsize
    for y in range(4):
        rows = ([(x+y)%4 for x in range(4)] + [4+(x+y)%4 for x in range(4)])
        d_ = np.linalg.norm(crnr2[rows, :] - crnr1, axis=1).sum() / 8.0            
        if d_ < dist:
            dist = d_

    u = sum([np.linalg.norm(x[0,:] - x[6,:]) for x in [crnr1, crnr2]])/2.0

    measure = max(1.0 - dist/u, 0)
    print(measure)
    
    
    return measure


def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths        
    """
    which_dim = len(points.shape) - 2 # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5*(mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)

def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """
    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[1,1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0,:] = np.array([cosval, 0, sinval])
        rotmat[2,:] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return
    
def corner2bbox(pcl, rot1):   ############################
  tp = []
  max0 = max(pcl[:,0])
  min0 = min(pcl[:,0])
  max1 = max(pcl[:,1])
  min1 = min(pcl[:,1])  
  max2 = max(pcl[:,2])
  min2 = min(pcl[:,2])  
  
  kk = 0.0*np.pi
  
  #print(pcl)
  
  if ((pcl[1,0]-pcl[0,0]) == 0.0):
    rot = 0
  else:
    rot = math.acos((pcl[1,0]-pcl[0,0])/(pcl[1,0]-pcl[0,0]))*np.pi

  temp1 = math.sqrt(pow(pcl[0,0]-pcl[4,0],2.0)+pow(pcl[0,1]-pcl[4,1],2.0))
  temp2 = math.sqrt(pow(pcl[0,0]-pcl[1,0],2.0)+pow(pcl[0,1]-pcl[1,1],2.0))  

  tp = [(max0+min0)/2.0,(max1+min1)/2.0,(max2+min2)/2.0,temp1/2.0,temp2/2.0,(max2-min2)/2.0]
  
  #print(tp)
  
  return tp, rot + kk
  
def updateAnno(obj,pcl_b,pcl_a,bbox_info,obj2ids):
  temp = []
  change2ids = {'add':0,'delete':1,'open':2,'close':3}
    
  if obj['type'] in ['add', 'delete']:
    rot = int(obj['location']['rotation']['y']) # ((int(obj['location']['rotation']['y']))/180.0)*np.pi
  else:
    rot = int(obj['location_after']['rotation']['y']) # ((int(obj['location_after']['rotation']['y']))/180.0)*np.pi    
    
  if obj['type'] in ['delete']:
    current_bbox, ro = corner2bbox(pcl_b, rot)
  else:
    current_bbox, ro = corner2bbox(pcl_a, rot)
 
  for b_ in current_bbox:
    temp.append(b_)

  temp.append(ro)   
  #print('rot',str(ro))   
  
  temp.append(change2ids[obj['type']])
  
  obj2cls = obj['obj'].split('|')[0]
  if obj2cls not in obj2ids:
    obj2ids[obj2cls] = len(obj2ids)
  temp.append(obj2ids[obj2cls])
  
  bbox_info.append(temp)

def downsamplePCD(before_ply, after_ply, pointnum):
  ###
  new_pcd = np.random.rand(pointnum,7) 
  new_pcd[0:int(pointnum/2),6] = 1
  new_pcd[int(pointnum/2):pointnum,6] = 2
  
  ##
  before_xyz = np.asarray(before_ply.points)
  before_rgb = np.asarray(before_ply.colors)
  choices_before = np.random.choice(before_xyz.shape[0], int(pointnum/2), replace=False)
  
  after_xyz = np.asarray(after_ply.points)
  after_rgb = np.asarray(after_ply.colors)
  choices_after = np.random.choice(after_xyz.shape[0],int(pointnum/2), replace=False)  
  
  new_pcd[0:int(pointnum/2),0:3] = before_xyz[choices_before]
  new_pcd[0:int(pointnum/2),3:6] = before_rgb[choices_before]
  new_pcd[int(pointnum/2):pointnum,0:3] = after_xyz[choices_after]
  new_pcd[int(pointnum/2):pointnum,3:6] = after_rgb[choices_after]
    
  return new_pcd
    
def computeAnno(before_ply,after_ply,change_info,lowest,bbox_info,obj2ids):
  ##
  before_ply = regulatePLY(before_ply)
  after_ply = regulatePLY(after_ply)  
  
  #o3d.io.write_point_cloud(ff+'/before_new.ply',before_ply)
  #o3d.io.write_point_cloud(ff+'/after_new.ply',after_ply)
  
  for obj in change_info:
    current_obj = obj['obj'] + '_' + obj['type']

    #for ooo in event.metadata["objects"]:
    #  if ooo['objectId'].split('|')[0] == obj['obj'].split('|')[0]:
    #    print(ooo)
    #exit()
    #print(event.metadata["objects"])
   
    if obj['type'] in ['add','delete']:
      cps = obj['location']['axisAlignedBoundingBox']['cornerPoints']

      pcl = get_bbox_from_corners(cps,lowest)  
      bbox = get_BBOXBoarder1(pcl)
      
      _, pnum_b = bboxChosePoint(before_ply, bbox)
      _, pnum_a = bboxChosePoint(after_ply, bbox)

      if (pnum_b > 100) and (pnum_a > 100):
        ###
        updateAnno(obj,pcl,pcl,bbox_info,obj2ids)
      
    else:
      cps_b = obj['location_before']['axisAlignedBoundingBox']['cornerPoints']
      cps_a = obj['location_after']['axisAlignedBoundingBox']['cornerPoints']
      
      pcl_b = get_bbox_from_corners(cps_b,lowest)   
      bbox_b = get_BBOXBoarder1(pcl_b)
      new_pcl_b, pnum_b = bboxChosePoint(before_ply, bbox_b)
      
      pcl_a = get_bbox_from_corners(cps_a,lowest)  
      bbox_a = get_BBOXBoarder1(pcl_a)
      new_pcl_a, pnum_a = bboxChosePoint(after_ply, bbox_a)                     
   
      if (pnum_b > 100) and (pnum_a > 100):
        ###
        updateAnno(obj,pcl_b,pcl_a,bbox_info,obj2ids)
    

  return before_ply, after_ply
    
def get_bbox_from_corners(bed,lowest):
  
  pcl = np.random.rand(8, 3)  
  for idx in range(len(bed)):
    pcl[idx] = [bed[idx][0], bed[idx][2], bed[idx][1]-lowest]
  
  return pcl
  
  """
  For plot bbox 
  pcl = np.random.rand(12, 2, 3)  
  pcl[0][0] = [bed[0][0],(bed[0][1]-lowest),bed[0][2]]
  pcl[0][1] = [bed[1][0],(bed[1][1]-lowest),bed[1][2]]
      
  pcl[1][0] = [bed[1][0],(bed[1][1]-lowest),bed[1][2]]
  pcl[1][1] = [bed[3][0],(bed[3][1]-lowest),bed[3][2]]


  pcl[2][0] = [bed[3][0],(bed[3][1]-lowest),bed[3][2]]
  pcl[2][1] = [bed[2][0],(bed[2][1]-lowest),bed[2][2]]
    
  pcl[3][0] = [bed[2][0],(bed[2][1]-lowest),bed[2][2]]
  pcl[3][1] = [bed[0][0],(bed[0][1]-lowest),bed[0][2]]
    
  pcl[4][0] = [bed[4][0],(bed[4][1]-lowest),bed[4][2]]
  pcl[4][1] = [bed[5][0],(bed[5][1]-lowest),bed[5][2]]    
    
  pcl[5][0] = [bed[5][0],(bed[5][1]-lowest),bed[5][2]]      
  pcl[5][1] = [bed[7][0],(bed[7][1]-lowest),bed[7][2]]        
    
  pcl[6][0] = [bed[7][0],(bed[7][1]-lowest),bed[7][2]]       
  pcl[6][1] = [bed[6][0],(bed[6][1]-lowest),bed[6][2]]        
    
  pcl[7][0] = [bed[6][0],(bed[6][1]-lowest),bed[6][2]] 
  pcl[7][1] = [bed[4][0],(bed[4][1]-lowest),bed[4][2]]
 
  pcl[8][0] = [bed[0][0],(bed[0][1]-lowest),bed[0][2]]  
  pcl[8][1] = [bed[4][0],(bed[4][1]-lowest),bed[4][2]]
            
  pcl[9][0] = [bed[2][0],(bed[2][1]-lowest),bed[2][2]]    
  pcl[9][1] = [bed[6][0],(bed[6][1]-lowest),bed[6][2]] 
    
  pcl[10][0] = [bed[1][0],(bed[1][1]-lowest),bed[1][2]]    
  pcl[10][1] = [bed[5][0],(bed[5][1]-lowest),bed[5][2]]  
    
  pcl[11][0] = [bed[3][0],(bed[3][1]-lowest),bed[3][2]]    
  pcl[11][1] = [bed[7][0],(bed[7][1]-lowest),bed[7][2]]   
  """
  pcl = np.random.rand(12, 2,3)  
  ###########################################3
  pcl[0][0] = [bed[0][0],bed[0][2],(bed[0][1]-lowest)]
  pcl[0][1] = [bed[1][0],bed[1][2],(bed[1][1]-lowest)]
      
  pcl[1][0] = [bed[1][0],bed[1][2],(bed[1][1]-lowest)]
  pcl[1][1] = [bed[3][0],bed[3][2],(bed[3][1]-lowest)]

  pcl[2][0] = [bed[3][0],bed[3][2],(bed[3][1]-lowest)]
  pcl[2][1] = [bed[2][0],bed[2][2],(bed[2][1]-lowest)]
    
  pcl[3][0] = [bed[2][0],bed[2][2],(bed[2][1]-lowest)]
  pcl[3][1] = [bed[0][0],bed[0][2],(bed[0][1]-lowest)]
    
  pcl[4][0] = [bed[4][0],bed[4][2],(bed[4][1]-lowest)]
  pcl[4][1] = [bed[5][0],bed[5][2],(bed[5][1]-lowest)]    
    
  pcl[5][0] = [bed[5][0],bed[5][2],(bed[5][1]-lowest)]      
  pcl[5][1] = [bed[7][0],bed[7][2],(bed[7][1]-lowest)]        
    
  pcl[6][0] = [bed[7][0],bed[7][2],(bed[7][1]-lowest)]       
  pcl[6][1] = [bed[6][0],bed[6][2],(bed[6][1]-lowest)]        
    
  pcl[7][0] = [bed[6][0],bed[6][2],(bed[6][1]-lowest)] 
  pcl[7][1] = [bed[4][0],bed[4][2],(bed[4][1]-lowest)]
 
  pcl[8][0] = [bed[0][0],bed[0][2],(bed[0][1]-lowest)]  
  pcl[8][1] = [bed[4][0],bed[4][2],(bed[4][1]-lowest)]
            
  pcl[9][0] = [bed[2][0],bed[2][2],(bed[2][1]-lowest)]    
  pcl[9][1] = [bed[6][0],bed[6][2],(bed[6][1]-lowest)] 
    
  pcl[10][0] = [bed[1][0],bed[1][2],(bed[1][1]-lowest)]    
  pcl[10][1] = [bed[5][0],bed[5][2],(bed[5][1]-lowest)]  
    
  pcl[11][0] = [bed[3][0],bed[3][2],(bed[3][1]-lowest)]    
  pcl[11][1] = [bed[7][0],bed[7][2],(bed[7][1]-lowest)]     
  
  
  return pcl
  
def get_BBOXBoarder(pcl):
  minX = 1000.0
  maxX = -1000.0
  minY = 1000.0
  maxY = -1000.0
  minZ = 1000.0
  maxZ = -1000.0  
  
  for pp in pcl:
    for p in pp:
      if p[0] < minX:
        minX = p[0]
      if p[0] > maxX:
        maxX = p[0]

      if p[1] < minY:
        minY = p[1]
      if p[1] > maxY:
        maxY = p[1]
        
      if p[2] < minZ:
        minZ = p[2]
      if p[2] > maxZ:
        maxZ = p[2]
        
  return [minX, maxX, minY, maxY, minZ, maxZ]
  
def get_BBOXBoarder1(pcl):
  return [min(pcl[:,0]), max(pcl[:,0]), min(pcl[:,1]), max(pcl[:,1]), min(pcl[:,2]), max(pcl[:,2])]

def bboxChosePoint(pcd, bbox):
  points = np.asarray(pcd.points)
  new_pcd = pcd.select_by_index(np.where((points[:,0] > bbox[0]) & (points[:,0] < bbox[1]) & (points[:,1] > bbox[2]) & (points[:,1] < bbox[3]) & (points[:,2] > bbox[4]) & (points[:,2] < bbox[5]))[0])
  
  new_pcd_point_number = len(np.asarray(new_pcd.points))
    
  return new_pcd, new_pcd_point_number
  
def regulatePLY(pc):
  xyz = np.asarray(pc.points)*1000.0
  rgb = pc.colors
  
  xyz[:,1] = -xyz[:,1]
  
  min_y = min(xyz[:,1])
  
  xyz[:,1] = xyz[:,1] - min_y

  new_xyz = np.zeros_like(xyz)
  
  new_xyz[:,0] = xyz[:,0]
  new_xyz[:,2] = xyz[:,1]
  new_xyz[:,1] = xyz[:,2]

  
  ### For visualization ###
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(new_xyz)
  pcd.colors = o3d.utility.Vector3dVector(rgb)
  
  
  """
  #o3d.visualization.draw_geometries([pcd])
  o3d.io.write_point_cloud(f_, pcd)
  """
  return pcd
    

def write_lines_as_cylinders(pcl, filename, color, rad=0.02, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos             
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    
    for src,tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0,0,1],vec, False)
        vec = tgt - src # compute again since align_vectors modifies vec in-place!
        M[:3,3] = 0.5*src + 0.5*tgt
        height = np.sqrt(np.dot(vec, vec))
        cylinder = trimesh.creation.cylinder(radius=rad, height=height, sections=res, transform=M)
        cylinder.visual.vertex_colors = color# trimesh.visual.random_color()#[200, 200, 250, 100]
        cylinder.visual.face_colors = color#trimesh.visual.random_color()#[200, 200, 250, 100]
        
        scene.add_geometry(cylinder)
        
    meshes = scene.dump()
    """
    meshes_out = []
    for mesh in meshes:
      mesh.visual.vertex_colors = trimesh.visual.random_color()
      mesh.visual.face_colors = trimesh.visual.random_color()
      meshes_out.append(mesh)
    """
    mesh_list = trimesh.util.concatenate(meshes)
    mesh_list.export('%s.ply'%(filename), file_type='ply')

"""
# ----------------------------------------
# Testing
# ----------------------------------------
if __name__ == '__main__':
    print('running some tests')
    
    ############
    ## Test "write_lines_as_cylinders"
    ############
    pcl = np.random.rand(32, 2, 3)
    write_lines_as_cylinders(pcl, 'point_connectors')
    input()
    
   
    scene_bbox = np.zeros((1,7))
    scene_bbox[0,3:6] = np.array([1,2,3]) # dx,dy,dz
    scene_bbox[0,6] = np.pi/4 # 45 degrees 
    write_oriented_bbox(scene_bbox, 'single_obb_45degree.ply')
    ############
    ## Test point_cloud_to_bbox 
    ############
    pcl = np.random.rand(32, 16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (32, 6)
    
    pcl = np.random.rand(16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)    
    assert pcl_bbox.shape == (6,)
    
    ############
    ## Test corner distance
    ############
    crnr1 = np.array([[2.59038660e+00, 8.96107932e-01, 4.73305349e+00],
 [4.12281644e-01, 8.96107932e-01, 4.48046631e+00],
 [2.97129656e-01, 8.96107932e-01, 5.47344275e+00],
 [2.47523462e+00, 8.96107932e-01, 5.72602993e+00],
 [2.59038660e+00, 4.41155793e-03, 4.73305349e+00],
 [4.12281644e-01, 4.41155793e-03, 4.48046631e+00],
 [2.97129656e-01, 4.41155793e-03, 5.47344275e+00],
 [2.47523462e+00, 4.41155793e-03, 5.72602993e+00]])
    crnr2 = crnr1

    print(bbox_corner_dist_measure(crnr1, crnr2))
    
    
    
    print('tests PASSED')
""" 
    

