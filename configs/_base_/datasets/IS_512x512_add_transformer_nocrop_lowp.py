# dataset settings
# import cv2

dataset_type = 'LEVIRCDDataset'
data_root = r'E:\lizhengcan\ISPRS-CD\train_set82'

albu_train_transforms = [
    # dict(type="RandomResizedCrop",height=1024,width=1024,interpolation=cv2.INTER_LINEAR,p=0.5),
    # dict(type="Resize", height=1024,width=1024,interpolation=2,p=1),
    # dict(type="RandomCrop",height=512,width=512,p=1),
    dict(type="CLAHE", p=0.2),
    dict(type='RandomBrightnessContrast', p=0.2),
    dict(type='RandomFog',p=0.2),
    dict(type="GaussianBlur",p=0.2),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
    dict(type='RandomRotate90',p=0.5)
]
train_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile',to_float32=False),
    dict(type='Load_IS_Annotations'),
    # dict(type="RandomHistogramMatching"), #直方图均衡化
    dict(
        type='Albu',
        keymap={
            'img': 'image',
            'img2': 'image2',
            'gt_seg_map': 'mask'
        },
        transforms=albu_train_transforms,
        additional_targets={'image2': 'image'},
        bgr_to_rgb=False),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile',to_float32=False),
    dict(type='Load_IS_Annotations'),
    # dict(type="RandomHistogramMatching"),  # 直方图均衡化
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]

# tta_pipeline = [
#     dict(type='Load_IS_MultipleRSImageFromFile',to_float32=False),
#     dict(
#         type='TestTimeAug',
#         transforms=[[dict(type='Load_IS_Annotations')],
#                     [dict(type='ConcatCDInput')],
#                     [dict(type='PackSegInputs')]])
# ]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/T1',
            img_path2='train/T2',
            seg_map_path='train/gt'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),

    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/T1', img_path2='test/T2', seg_map_path='test/gt'),
        pipeline=test_pipeline),

)
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator