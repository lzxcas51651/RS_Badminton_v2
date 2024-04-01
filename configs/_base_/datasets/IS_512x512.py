# dataset settings
dataset_type = 'LEVIRCDDataset'
data_root = r'E:\lizhengcan\ISPRS-CD\train_set102'

albu_train_transforms = [
    dict(type='RandomBrightnessContrast', p=0.2),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]
train_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile'),
    dict(type='Load_IS_Annotations'),
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
    dict(type='Load_IS_MultipleRSImageFromFile'),
    dict(type='Load_IS_Annotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='Load_IS_MultipleRSImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[dict(type='Load_IS_Annotations')],
                    [dict(type='ConcatCDInput')],
                    [dict(type='PackSegInputs')]])
]
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