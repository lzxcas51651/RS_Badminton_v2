_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/IS_512x512-test.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k-test.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])
# checkpoint_file ='https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),#如果要使用预训练模型，要在backbone中加这一行代码。
        in_channels=6,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    auxiliary_head=dict(in_channels=384, num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
#
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=20000,
#         by_epoch=False,
#     )
# ]

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
