# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseCDDataset


@DATASETS.register_module()
class LEVIRCDDataset(BaseCDDataset):
    """ISPRS dataset.

    In segmentation map annotation for ISPRS, 0 is to ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    # METAINFO = dict(
    #     classes=('background', 'changed'),
    #     palette=[[0, 0, 0], [255, 255, 255]])
    METAINFO = dict(
        classes=('background', 'garden', 'pond', 'soil', 'building', 'railway', 'road'),
        palette=[[0, 0, 0], [50, 50, 50], [100, 100, 100], [150, 150, 150], [200, 200, 200], [220, 220, 220],
                 [250, 250, 250]])

    def __init__(self,
                 img_suffix='第二期影像.tif',
                 img_suffix2='第三期影像.tif',
                 seg_map_suffix='.tif',
                 # img_suffix='.png',
                 # img_suffix2='.png',
                 # seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
