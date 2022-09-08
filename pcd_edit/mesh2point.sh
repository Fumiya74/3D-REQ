#!/bin/sh

dir_path="/home/fumiya/matsu/3RScan/data/3RScan/*"
#dirs= find $dir_path -maxdepth 0 -type d

for dir in $(find $dir_path -maxdepth 0 -type d);
do
    echo $dir
    cd $dir
    cloudcompare.CloudCompare -SILENT -O mesh.refined.v2.obj -SAMPLE_MESH POINTS 80000 -C_EXPORT_FMT PLY -PLY_EXPORT_FMT ASCII -NO_TIMESTAMP -SAVE_CLOUDS
done
