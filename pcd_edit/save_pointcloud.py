import open3d as o3d
import numpy as np
import os
from glob import glob
from tqdm import tqdm
pointnum = 50000
path = "/home/fumiya/matsu/3RScan/data/3RScan/"
output = "/home/fumiya/matsu/CyREx/data/datasets/pcl"
all_scans = glob(os.path.join(path,"*"))

for scan in tqdm(all_scans):

    #print(os.path.join(scan,'mesh.refined.v2_SAMPLED_POINTS.ply'))
    #aps = o3d.io.read_point_cloud('/home/fumiya/matsu/3RScan/Scans/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/labels.instances.annotated.v2.ply')
    aps = o3d.io.read_point_cloud(os.path.join(scan,'mesh.refined.v2_SAMPLED_POINTS.ply'))
    new_pcd = np.random.rand(pointnum,6) 

    aps_xyz = np.asarray(aps.points)
    aps_rgb = np.asarray(aps.colors)
    choices = np.random.choice(aps_xyz.shape[0], pointnum, replace=False)

    new_pcd[:,0:3] = aps_xyz[choices]
    new_pcd[:,3:6] = aps_rgb[choices]
    if new_pcd.shape != (50000,6):
        print("Size Error")

    #Convert sampled point cloud to .ply format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_pcd[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(new_pcd[:,3:6])
    o3d.io.write_point_cloud(os.path.join(scan,"test.ply"), pcd)
    #output + "/" + os.path.basename(scan)
    #print(output + "/" + os.path.basename(scan))
    #np.savez(output + "/" + os.path.basename(scan), pc=new_pcd)