_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])
model = dict(data_preprocessor=data_preprocessor,
             backbone=dict(
                 in_channels=6,),
             decode_head=dict(
                 num_classes=2),
             auxiliary_head=dict(
                 num_classes=2)
             )
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
