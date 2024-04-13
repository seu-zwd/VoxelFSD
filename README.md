# VoxelFSD
![“Fig. 1 pipeline of VoxxelFSD“](pic/model.jpg)

This repository is the official code of the paper "VoxelFSD: voxel-based fully sparse detector for 3D object detection", VoxelFSD is a voxel-based fully sparse detector, the workflow is shown above, which has significant real-time performance on large-scale point clouds compared with the previous voxel-based methods. Furthermore, VoxelFSD-S get `77.67`map for car class on `KITTI`dataset, and VoxelFSD-T further reach `81.50`, which is competitive. 

## Results
||car@Easy|car@Mod.|car@Hard||
|---|---|---|---|---|
|VoxelFSD-S|77.67|86.29|72.18|[download](https://pan.baidu.com/s/1PuTBm4rSQ6HvkrzgEdO4sg?pwd=1234)|
|VoxelFSD-T|89.89|81.50|76.82|[download](https://pan.baidu.com/s/14hwOdXIwMWpOy7eg6dKdOQ?pwd=1234)|

<div style="text-align:left;">
    <img src="pic/time.png" alt="Image" width="800" height="600">
</div>


## Installation
* Download the `KITTI` dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
* Prepare the data as `pcdet` did in [data prepare](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)
* Install this pcdet library and its dependent libraries by running the following command:
```python
python setup.py develop
```

## Start
* train<br>
first run  `cd tools` in terminal and than run
```python
python train.py --cfg_file tools/VoxelFSD-S.yaml // for VoxelFSD-S
python train.py --cfg_file tools/VoxelFSD-T.yaml // for VoxelFSD-T
```
* test<br>
```python
python test.py --cfg_file tools/VoxelFSD-S.yaml --ckpt path/to/your/model // for VoxelFSD-S
python test.py --cfg_file tools/VoxelFSD-S.yaml --ckpt path/to/your/model // for VoxelFSD-T
```
##Citation
out work is based on the [pcdet](https://github.com/open-mmlab/OpenPCDet?tab=readme-ov-file)
