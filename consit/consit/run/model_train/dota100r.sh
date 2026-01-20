#dota的开关
# for voc, copy the initial pseudo-labels to semivoc dir
rm -rf /home/data/szc/remotedsl/data/semivoc/unlabel_prepared_annos/Industry/annotations/full2_dota10/
#cp -r workdir_voc/RLA_r50_caffe_mslonger_tricks_alldata/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/

# 进行修改，需要使用绝对路径进行计算
cp -r /home/data/szc/remotedsl/DSL/workdir_voc/r50_caffe_mslonger_tricks_07data_built_in/epoch_60.pth-unlabeled.bbox.json_thres0.1_annos/  /home/data/szc/remotedsl/data/semivoc/unlabel_prepared_annos/Industry/annotations/full2_dota10/
echo "remove & copy annotations done!"

#CONFIG=configs/fcos_semi/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
CONFIG=configs/fcos_semi/voc/dota100.py
#WORKDIR=workdir_coco/sunzicheng
WORKDIR=workdir_voc/fuxiantrain2_dota100_p0.7
GPU=2

CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR
#CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR --resume-from  /home/data/szc/consit/consit/workdir_voc/train2_dota10/epoch_25.pth