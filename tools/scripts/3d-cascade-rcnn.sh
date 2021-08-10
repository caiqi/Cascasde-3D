#!/usr/bin/env bash
NGPUS=4
CFG_NAME=3d_cascade_rcnn
python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 2


python test.py --cfg_file cfgs/3d_cascade_rcnn.yaml --workers 0 --ckpt /export/home/v-qcaii/research/hydrogen/Voxel-R-CNN/output/kitti_results/validation/checkpoint_epoch_51.pth
