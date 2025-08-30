#!/bin/bash

cd <ROOT>/DFC2019/indoor_segmentation/script

### Path
PU_GAN_MESH=/root/dataset/PU-GAN
SCANNET_TRAIN=/root/dataset/deepmvs/train
SCANNET_TEST=/root/dataset/deepmvs/test
SCANNET_SEMSEG=/root/dataset/scannet_semseg
SHAPENET=/root/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/
SHAPNETCORE=/root/dataset/shapenetcore/ShapeNetCore.v2/
S3DIS=./DFC2019_test # your dataset path

### Setup 
MYSHELL="test.sh"
DATE_TIME=`date +"%Y-%m-%d"`
NEPTUNE_PROJ="ychen1112/YchenEPCL"
COMPUTER="S3DIS-EPCL-00"
export MASTER_ADDR='localhost'
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0

### Params
WORKERS=4
NUM_GPUS=1
NUM_TRAIN_BATCH=8
NUM_VAL_BATCH=8
NUM_TEST_BATCH=8

ARCH="pointtransformer"
DATASET="loader_s3dis"
INTRALAYER="PointTransformerIntraSetLayer"
INTERLAYER="NoInterSetLayer"
TRANSDOWN="TransitionDownBlock"
TRANSUP="TransitionUp"

MYCHECKPOINT="<ROOT>/DFC2019/indoor_segmentation/checkpoints"

cd ../

## TEST (pre-process stage for test dataset) 
## npts=800k
## TEST (pre-process stage for test dataset)
## npts=800k

## TEST (evaluation)
## npts=800k
python test_pl.py \
  --MYCHECKPOINT $MYCHECKPOINT --computer $COMPUTER --shell $MYSHELL \
  --MASTER_ADDR $MASTER_ADDR \
  --train_worker $WORKERS --val_worker $WORKERS \
  --NUM_GPUS $NUM_GPUS  \
  --train_batch $NUM_TRAIN_BATCH  \
  --val_batch $NUM_VAL_BATCH  \
  --test_batch $NUM_TEST_BATCH \
  \
  --scannet_train_root $SCANNET_TRAIN  --scannet_test_root $SCANNET_TEST \
  --scannet_semgseg_root $SCANNET_SEMSEG \
  --shapenet_root $SHAPENET  --shapenetcore_root $SHAPNETCORE \
  --s3dis_root $S3DIS \
  \
  --neptune_proj $NEPTUNE_PROJ \
  --epochs 64  --CHECKPOINT_PERIOD 1  --lr 0.1 \
  --dataset $DATASET --optim 'SGD' \
  \
  --model 'net_epcl' --arch $ARCH  \
  --intraLayer $INTRALAYER  --interLayer $INTERLAYER \
  --transdown  $TRANSDOWN --transup $TRANSUP \
  --nsample 8 16 16 16 16  --drop_rate 0.1  --fea_dim 5  --classes 5 \
  \
  --voxel_size 1  --eval_voxel_max 800000  --test_batch 8
#####

cd -