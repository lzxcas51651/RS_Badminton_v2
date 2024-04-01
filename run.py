import glob
import os.path
import os.path as osp
from argparse import ArgumentParser
from unittest import TestCase

import numpy as np
from mmengine import ConfigDict, init_default_scope
# from utils import *  # noqa: F401, F403

from mmseg.apis import RSImage, RSInferencer
from mmseg.apis.remote_sense_inferencer import RSInferencer_twoImage
from mmseg.registry import MODELS

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)


def test_read_and_inference():
    init_default_scope('mmseg')
    cfg_dict = dict(
        checkpoint_path=r'E:\lizhengcan\ISPRS-CD\mmsegmentation-fix_cd_transform\experiment\ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512-ALL\iter_80000.pth',

        model=dict(
            type='EncoderDecoder',
            data_preprocessor=data_preprocessor,
            backbone=dict(
                type='SwinTransformer',
                in_channels=6,
                pretrain_img_size=224,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg),
            decode_head=dict(
                type='UPerHead',
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=7,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=384,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=7,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole')),
        test_dataloader=dict(
            dataset=dict(
                type='LEVIRCDDataset',
                pipeline=[
                    dict(type='Load_IS_MultipleRSImageFromFile'),
                    # dict(type='Load_IS_Annotations'),
                    # dict(type='ConcatCDInput'),
                    dict(type='PackSegInputs')
                ])),
        test_pipeline=[
            dict(type='Load_IS_MultipleRSImageFromFile'),
            # dict(type='Load_IS_Annotations'),
            # dict(type='ConcatCDInput'),
            dict(type='PackSegInputs')
        ])


    METAINFO = dict(
        classes=('background', 'garden', 'pond', 'soil', 'building', 'railway', 'road'),
        palette=[[0, 0, 0], [50, 50, 50], [100, 100, 100], [150, 150, 150], [200, 200, 200], [220, 220, 220],
                 [250, 250, 250]])
    cfg = ConfigDict(cfg_dict)
    model = MODELS.build(cfg.model)
    model.cfg = cfg
    inferencer = RSInferencer_twoImage.from_model(model,checkpoint_path=cfg.checkpoint_path,thread=4,device='cuda:0')

    parser = ArgumentParser()
    # parser.add_argument('in_dir', help='in_dir path')
    # parser.add_argument('out_dir', help='out_dir path')
    parser.add_argument('--in_dir', help='in_dir path',default=r"E:\lizhengcan\ISPRS-CD\train_set82\train")
    parser.add_argument('--out_dir', help='out_dir path',default=r"E:\lizhengcan\ISPRS-CD\train_set82\train\out")
    args = parser.parse_args()

    test_dir=args.in_dir
    out_dir=args.out_dir
    # print(test_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    T1_dir=os.path.join(test_dir,"T1")
    T2_dir=os.path.join(test_dir,"T2")
    img_list = glob.glob(T1_dir + '/*.tif')
    for img_path in img_list:
        T1_img_path=img_path
        T2_img_path=img_path.replace("T1","T2").replace("二","三")
        img_name_base = os.path.basename(img_path).replace("第二期影像","")
        out_img_path=os.path.join(out_dir,img_name_base)

        T1_img = RSImage(T1_img_path)
        T2_img = RSImage(T2_img_path)

        inferencer.run(T1_img,T2_img,METAINFO,out_img_path,False)


        # print(img_path)

if __name__ == '__main__':
    test_read_and_inference()
