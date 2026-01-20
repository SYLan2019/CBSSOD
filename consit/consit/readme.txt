清单

    文件夹介绍
        位于八卡服务器目录/public/home/lsy/myj
        consit文件夹下


    实验环境
        实验关键虚拟环境配置：
        pytorch                   1.10.1          py3.9_cuda11.3_cudnn8.2.0_0    pytorch
        pytorch-mutex             1.0                        cuda    pytorch
        mmcv                      1.3.10                   pypi_0    pypi
        mmcv-full                 1.4.0                    pypi_0    pypi
        opencv                    4.7.0            py39hf3d152e_0    conda-forge
        opencv-python             4.7.0.72                 pypi_0    pypi
        cudatoolkit               11.3.1              h9edb442_11    conda-forge
        torchaudio                0.10.1               py39_cu113    pytorch
        torchvision               0.11.2               py39_cu113    pytorch

    实验内容：
        运行脚本：
        ./run/model_train/dota100.sh
            DOTA相关：
                1%：\consit\consit\run\model_train\dota01.sh
                5%：\consit\consit\run\model_train\dota05.sh
                10%：\consit\consit\run\model_train\dota10.sh
                100%：
                    基础检测器（burn in阶段）：
                        \consit\consit\run\model_train\baseline_voc.sh
                    半监督总模型：
                        \consit\consit\run\model_train\dota100.sh


            NWPU相关：
                consit/mmdet/datasets/semivoc.py:32 打开NWPU类别
                consit/mmdet/datasets/semivoc.py:96 打开NWPU数据集的读取图片后缀
                consit/mmdet/datasets/voc.py:29 打开nwpu类别
                consit/mmdet/datasets/xml_style.py:72 打开NWPU数据集的读取图片后缀
                30%：\consit\consit\run\model_train\nwpu30.sh
                40%：\consit\consit\run\model_train\nwpu40.sh
                50%：\consit\consit\run\model_train\nwpu50.sh
                100：
                    基础检测器（burn in阶段）：
                        consit/configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py:27 改为10表示nwpu种类数
                        consit/configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py:111 打开
                        consit/configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py:125 打开
                        consit/configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py:136 打开
                        consit/run/model_train/baseline_voc.sh:19 换成workdir_voc/nwpu100_burn_in
                        #脚本运行
                        \consit\consit\run\model_train\baseline_voc.sh
                    半监督总模型：
                        \consit\consit\run\model_train\nwpu100.sh


            消融相关：
                增强相关：consit/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:108
                rla骨干：consit/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:4
                kl损失：consit/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:258
                全局损失： consit/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:277