_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/IS_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),#如果要使用预训练模型，要在backbone中加这一行代码。
        in_channels=6),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
