#!/usr/bin/env bash
NGPUS=4
CFG_NAME=3d_cascade_rcnn
python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 2 --ckpt ../checkpoints/3d_cascade_rcnn.pth