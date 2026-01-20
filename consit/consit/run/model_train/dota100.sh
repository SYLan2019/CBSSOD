#dota的开关
# for voc, copy the initial pseudo-labels to semivoc dir
#下面这段代码的意思是删除semivoc目录下的full2_dota100目录
rm -rf /home/scdx2/szc/consit/data/semivoc/unlabel_prepared_annos/Industry/annotations/full2_dota100/
#cp -r workdir_voc/RLA_r50_caffe_mslonger_tricks_alldata/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/

# 进行修改，需要使用绝对路径进行计算
cp -r /home/scdx2/szc/consit/consit/workdir_voc/fulldota/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/  /home/scdx2/szc/consit/data/semivoc/unlabel_prepared_annos/Industry/annotations/full2_dota100/
echo "remove & copy annotations done!"



#CONFIG=configs/fcos_semi/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
CONFIG=configs/fcos_semi/voc/dota100.py
#WORKDIR=workdir_coco/sunzicheng
WORKDIR=workdir_voc/train2_dota100
# 从第25个epoch的权重继续训练（如需使用 EMA 权重可改为 "$WORKDIR/epoch_25.pth_ema"）
RESUME=$WORKDIR/epoch_25.pth

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE=$WORKDIR/train_$TIMESTAMP.log
mkdir -p $WORKDIR
exec > >(tee -a "$LOGFILE") 2>&1

GPU=2

CUDA_VISIBLE_DEVICES=1,2 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR --cfg-options load_from=$RESUME optimizer.lr=0.000001