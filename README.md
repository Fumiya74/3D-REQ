# 3D-REQ:Question Generation for Uncertainty Elimination of Referring Expression in 3D Environment

## Project 
``` 
┣- 3RScan/
┣- data/
  ┗ datasets/
   ┣ train/val/test
   ┣ pcl/
   ┗ id2scene_ref.json
┣- pcd_edit/
┣- ssg2req
  ┗ scripts/
```
## How to Running Code
### Sumpling Point Clouds from 3RScan
```
cd pcd_edit
bash ./mash2point.sh
python save_pointcloud.py

```
### Generating Referring Expressions and Qestions from 3DSSG
```
cd ssg2req/scripts
python main.py
python format.py
```
### 
