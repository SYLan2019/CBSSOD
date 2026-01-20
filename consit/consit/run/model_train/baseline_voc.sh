#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
#下面是dota数据集的设置，主要要设置配置文件里面的参数，注意改变里面的日志文件
#CONFIG=configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py
#WORKDIR=workdir_voc/r50_caffe_mslonger_tricks_07data_built_in_0.05
#GPU=2
#
#CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR


CONFIG=configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py
WORKDIR=workdir_voc/dota100_burn_in
GPU=2

CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR
