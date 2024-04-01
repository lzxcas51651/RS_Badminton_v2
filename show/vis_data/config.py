albu_train_transforms = [
    dict(p=0.2, type='RandomBrightnessContrast'),
    dict(p=0.5, type='HorizontalFlip'),
    dict(p=0.5, type='VerticalFlip'),
]
backbone_norm_cfg = dict(requires_grad=True, type='LN')
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'E:\\lizhengcan\\ISPRS-CD\\train_set102'
dataset_type = 'LEVIRCDDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '../experiment\\ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512-ALL-train2500-test500\\iter_64000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=384,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=96,
        in_channels=6,
        mlp_ratio=4,
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        patch_size=4,
        pretrain_img_size=224,
        qk_scale=None,
        qkv_bias=True,
        strides=(
            4,
            2,
            2,
            2,
        ),
        type='SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    pretrained=
    'E:\\lizhengcan\\ISPRS-CD\\mmsegmentation-fix_cd_transform\\experiment\\ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512-ALL\\iter_80000.pth',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
    dict(
        begin=20000,
        by_epoch=False,
        end=80000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='test/T1', img_path2='test/T2', seg_map_path='test/gt'),
        data_root='E:\\lizhengcan\\ISPRS-CD\\train_set102',
        pipeline=[
            dict(type='Load_IS_MultipleRSImageFromFile'),
            dict(type='Load_IS_Annotations'),
            dict(type='ConcatCDInput'),
            dict(type='PackSegInputs'),
        ],
        type='LEVIRCDDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    keep_results=True,
    output_dir='../out',
    type='IoUMetric')
test_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile'),
    dict(type='Load_IS_Annotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='train/T1', img_path2='train/T2',
            seg_map_path='train/gt'),
        data_root='E:\\lizhengcan\\ISPRS-CD\\train_set102',
        pipeline=[
            dict(type='Load_IS_MultipleRSImageFromFile'),
            dict(type='Load_IS_Annotations'),
            dict(
                additional_targets=dict(image2='image'),
                bgr_to_rgb=False,
                keymap=dict(gt_seg_map='mask', img='image', img2='image2'),
                transforms=[
                    dict(p=0.2, type='RandomBrightnessContrast'),
                    dict(p=0.5, type='HorizontalFlip'),
                    dict(p=0.5, type='VerticalFlip'),
                ],
                type='Albu'),
            dict(type='ConcatCDInput'),
            dict(type='PackSegInputs'),
        ],
        type='LEVIRCDDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile'),
    dict(type='Load_IS_Annotations'),
    dict(
        additional_targets=dict(image2='image'),
        bgr_to_rgb=False,
        keymap=dict(gt_seg_map='mask', img='image', img2='image2'),
        transforms=[
            dict(p=0.2, type='RandomBrightnessContrast'),
            dict(p=0.5, type='HorizontalFlip'),
            dict(p=0.5, type='VerticalFlip'),
        ],
        type='Albu'),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile'),
    dict(
        transforms=[
            [
                dict(type='Load_IS_Annotations'),
            ],
            [
                dict(type='ConcatCDInput'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='test/T1', img_path2='test/T2', seg_map_path='test/gt'),
        data_root='E:\\lizhengcan\\ISPRS-CD\\train_set102',
        pipeline=[
            dict(type='Load_IS_MultipleRSImageFromFile'),
            dict(type='Load_IS_Annotations'),
            dict(type='ConcatCDInput'),
            dict(type='PackSegInputs'),
        ],
        type='LEVIRCDDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='../show',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '../experiment_test\\ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512-ALL-train2500-test500.'
