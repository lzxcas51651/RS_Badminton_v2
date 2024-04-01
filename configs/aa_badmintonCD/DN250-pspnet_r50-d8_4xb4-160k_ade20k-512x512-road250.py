_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/IS_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    pretrained="E:\lizhengcan\ISPRS-CD\mmsegmentation-fix_cd_transform\experiment\DN250-pspnet_r50-d8_4xb4-160k_ade20k-512x512-road250\iter_72000.pth",
    backbone=dict(
        type='ResNetV1c',
        in_channels=6,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),),
    decode_head=dict(
        num_classes=2),
    auxiliary_head=dict(
        num_classes=2))
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
