# CyREx(Clarify your Referring Expression):Dialog Generation for Uncertainty Elimination in 3D Grounding

##Project
```
┣-3RScan/
┣-data/
  ┗datasets/
   ┣train/val/test
   ┣pcl/
   ┗id2scene_ref.json
┣-pcd_edit/
┣-ssg2req
  ┗scripts/
```
## How to Running Code
### Sumpling Point Clouds from 3RScan
```
cd pcd_edit
bash .mash2point.sh
python save_pointcloud.py

```
### Generating Referring Expressions and Qestions from 3DSSG
```
cd ssg2req/scripts
python main.py
python format.py
```
### 
