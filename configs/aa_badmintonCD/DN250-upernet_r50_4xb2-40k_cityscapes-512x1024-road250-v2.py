_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/IS_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])
model = dict(data_preprocessor=data_preprocessor,
             pretrained="E:\lizhengcan\ISPRS-CD\mmsegmentation-fix_cd_transform\experiment\DN250-upernet_r50_4xb2-40k_cityscapes-512x1024-road250\iter_40000.pth",

             backbone=dict(
                 in_channels=6,),
             decode_head=dict(
                 num_classes=2),
             auxiliary_head=dict(
                 num_classes=2)
             )

# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=20000,
        end=80000,
        by_epoch=False)

]

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
