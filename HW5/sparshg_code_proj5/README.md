# Assignment 5 Point Cloud Processing

## Instructions to run this code

1. To train classification model of PointNet, CLI command is: python3 train.py --task cls

2. To train segmentation model of PointNet, CLI command is: python3 train.py --task seg

Note that visualization code has been embedded in `utils.py`, `eval_cls.py`, and `eval_seg.py`

3. Robustness Analysis doesn't have separate script. Rotation variation code has been commented out in `eval_cls.py`, and `eval_seg.py`. And number of points can be varied through argument --num_points

4. PointNet ++ has been implemented in `pointnet_pp.py`
Point Transformer has been implemented in `point_trans.py`