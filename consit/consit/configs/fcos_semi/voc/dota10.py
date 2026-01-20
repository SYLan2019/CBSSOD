model = dict(
    type='FCOS',
    backbone=dict(
        type='RLA_ResNet',
        layers=[3,4,6,3],
        #depth=50,
        #num_stages=4,
        #out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        #norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='/home/data/szc/remotedsl/resnet50_rla_2283.pth.tar'),
        # type='ResNet',
        # depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=False),
        # norm_eval=True,
        # style='caffe',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/public/home/lsy/myj/code_dsl/resnet50_msra-5891d200.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level = 0,
        #start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        #12.24进行消融实验
        type='FCOSHead',
        #种类数目一定需要修改
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # myj增加一个属性，这个属性用来判断是否增加richness头
        #richness=True,
        #strides=[8, 16, 32, 64, 128],
        strides=[4, 8, 16, 32, 64],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        #中心采样关
        center_sampling=False,
        conv_bias=True,
        # for unlabel loss weight
        loss_weight = 2.0,
        soft_weight = 1.0,
        soft_warm_up = 0,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0),#修改为ciou损失，更加注意分布相似性
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # assigner=dict(
        #     type='HieAssigner',
        #      ignore_iof_thr=-1,
        #      gpu_assign_thr=256,
        #      iou_calculator=dict(type='BboxDistanceMetric'),
        #      assign_metric='kl',
        #      topk=[3,1],
        #      ratio=0.9)
             ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='Resize',
    #     img_scale=[(1000, 480), (1000, 600)],
    #     multiscale_mode='value',
    #     keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1000, 736), (1000, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='PatchShuffle', ratio=0.5, ranges=[0.0,1.0], mode=['flip','flop']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], meta_keys=['filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'scale_idx', 'flip','flip_direction', 'img_norm_cfg', 'PS', 'PS_place', 'PS_mode']),
    #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], meta_keys=['filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'scale_idx', 'flip','flip_direction', 'img_norm_cfg']),
]
unlabel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='Resize',
    #     img_scale=[(1000, 480), (1000, 600)],
    #     multiscale_mode='value',
    #     keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1000, 736), (1000, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='PatchShuffle', ratio=0.5, ranges=[0.0,1.0], mode=['flip','flop']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAugmentBBox_Fast', aug_type='affine'),
    dict(type='UBAug'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], meta_keys=['filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'scale_idx', 'flip','flip_direction', 'img_norm_cfg', 'PS', 'PS_place', 'PS_mode']),
    #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], meta_keys=['filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'scale_idx', 'flip','flip_direction', 'img_norm_cfg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'SemiVOCDataset'
## consit style path; recommend to use absolute path
data_root = '/home/data/szc/remotedsl/data/semivoc/'

data = dict(
    #训练batch_size有改动，不能进行修改
    samples_per_gpu=2,
    workers_per_gpu=2,
    batch_config=dict(ratio =[[1, 1],]),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'prepared_annos/Industry/train_list1.txt',
        ann_path=data_root + 'prepared_annos/Industry/annotations/full/',
        labelmapper = data_root + 'mmdet_category_info.json',
        img_prefix = data_root + 'images/full/',
        pipeline = train_pipeline,
        ),
    unlabel_train=dict(
        type=dataset_type,
        ann_file=data_root + 'unlabel_prepared_annos/Industry/voc12_trainval.txt',
        ann_path=data_root + 'unlabel_prepared_annos/Industry/annotations/full2_dota10/',
        labelmapper = data_root + 'mmdet_category_info.json',
        img_prefix = data_root + 'unlabel_images/full/',
        pipeline = unlabel_train_pipeline,
        #myj 需要修改阈值，避免造成污染
        thres="adathres_2_dota10.json",
        ),
    unlabel_pred=dict(
        type=dataset_type,
        #myj 增加一个属性s
        #11.27进行修改,CBAM进行修改,12.24,在伪标记生成阶段生成boost值的开关
        richness=False,
        #这里也有待商榷
        num_gpus = 2,
        image_root_path = data_root + "unlabel_images/full/",
        #这个文件有待商榷******************************************
        #image_list_file = 'data_list/voc_semi/voc12_trainval.json',
        #myj 2023.6.6 读取未标记图像列表文件
        image_list_file = '/home/data/szc/remotedsl/data/semivoc/unlabel_prepared_annos/Industry/voc12_trainval.txt',
        anno_root_path = data_root + 'unlabel_prepared_annos/Industry/annotations/full2_dota10/',
        category_info_path = data_root + 'mmdet_category_info.json',
        infer_score_thre=0.1,
        save_file_format="json",
        pipeline = test_pipeline,
        eval_config ={"iou":[0.6]},
        img_path = data_root + "unlabel_images/full/",
        img_resize_size = (1000,800),
        low_level_scale = 16,
        use_ema=True,
        eval_flip=False,
        fuse_history=False,
        first_fuse=False,
        first_score_thre=0.1,
        eval_checkpoint_config=dict(interval=1, mode="iteration"),
        # 2*num_worker+2
        preload=6,
        start_point=8),
    #### For evaluate the VOC metric, metric="mAP"
    val=dict(
        type='VOCDataset',
        ann_file='/home/data/szc/remotedsl/ori_data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt',
        img_prefix='/home/data/szc/remotedsl/ori_data/VOCdevkit2007/VOC2007/',
        pipeline=test_pipeline),
    #### For evaluate the COCO metric, metric="bbox"
    #val=dict(
    #    type='Voc2CocoDataset',
    #    ann_file='data_list/voc_semi/voc07_test.json',
    #    img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/JPEGImages/',
    #    pipeline=test_pipeline),

    #### For inferencing pseudo-labels of the unlabel images via tools/inference_unlabeled_coco_data.sh
    # test=dict(
    #     type='Voc2CocoDataset',
    #     ann_file='data_list/voc_semi/voc12_trainval.json',
    #     img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2012/JPEGImages/',
    #     pipeline=test_pipeline)
    test=dict(
        type='VOCDataset',
        ann_file='/home/data/szc/remotedsl/ori_data/VOCdevkit2007/VOC2012/ImageSets/Main/unlabel_train.txt',
        img_prefix='/home/data/szc/remotedsl/ori_data/VOCdevkit2007/VOC2012',
        pipeline=test_pipeline)
)
# learning policy
optimizer = dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0001,paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    #_delete_=True,
    grad_clip=dict(max_norm=10, norm_type=2))
    #grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    #myj 2023.6.6 进行修改，大概总迭代次数1/3的位置自动修改学习率
    warmup_iters=1500,
    warmup_ratio=1.0 / 3,
    step=[20, 26])
#myj 12.5对最大max_epochs进行台修改
runner = dict(type='SemiEpochBasedRunner', max_epochs=28)
### VOC metric use "mAP", COCO metric use "bbox"
evaluation = dict(interval=1, metric='mAP')
#myj 2023.6.5 修改为每5个epoch迭代一次
checkpoint_config = dict(interval=5)
ema_config = dict(interval=1, mode="iteration",ratio=0.99,start_point=1)
scale_invariant = True

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
#myj 2023.6.6 你
load_from = None
resume_from = None
workflow = [('train', 1)]
#myj 修改，现在要试试richness头的效果，暂时不用特征融合，因此进行false
down_fuse = True
find_unused_parameters=True