# swin test0
config = ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512-ALL.py

data_root ='E:\\lizhengcan\\ISPRS-CD\\train_set82'

pretrained = 
'E:\\lizhengcan\\ISPRS-CD\\mmsegmentation-fix_cd_transform\\experiment\\swin-tiny-patch4-window7_upernet_1xb8-80k_levir-1024x1024\\iter_8000.pth',

增强：
```
dict(p=0.2, type='RandomBrightnessContrast'),
dict(p=0.5, type='HorizontalFlip'),
dict(p=0.5, type='VerticalFlip'),
```
40k Iou
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 96.25 | 99.54 |
|   garden   |  3.8  |  4.04 |
|    pond    | 17.27 | 20.73 |
|    soil    | 31.51 | 38.31 |
|  building  | 17.21 |  19.4 |
|  railway   |  0.0  |  0.0  |
|    road    | 10.55 |  11.2 |
+------------+-------+-------+
2024/03/25 23:07:00 - mmengine - INFO - Iter(val) [500/500]    aAcc: 96.0200  mIoU: 25.2300  mAcc: 27.6000  data_time: 0.0013  time: 0.0191
```
80k Iou
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 96.26 |  99.4 |
|   garden   |  2.19 |  2.3  |
|    pond    | 20.17 | 27.84 |
|    soil    | 30.12 |  34.8 |
|  building  | 21.53 | 26.23 |
|  railway   |  4.26 |  4.77 |
|    road    | 21.14 | 25.47 |
+------------+-------+-------+
2024/03/26 00:53:19 - mmengine - INFO - Iter(val) [500/500]    aAcc: 96.0300  mIoU: 27.9500  mAcc: 31.5400  data_time: 0.0013  time: 0.0192
```

# swin test1

config = ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512_Crop_CLANE_Resize_Contrast_H_V.py

data_root ='E:\\lizhengcan\\ISPRS-CD\\train_set82'

pretrained = 'E:\\lizhengcan\\ISPRS-CD\\mmsegmentation-fix_cd_transform\\experiment\\swin-tiny-patch4-window7_upernet_1xb8-80k_levir-1024x1024\\iter_8000.pth'

增强：

```
dict(
    height=1024,
    interpolation=2,
    p=1,
    type='Resize',
    width=1024),
dict(height=512, p=1, type='RandomCrop', width=512),
dict(p=0.5, type='CLAHE'),
dict(p=0.2, type='RandomBrightnessContrast'),
dict(p=0.5, type='RandomFog'),
dict(p=0.5, type='GaussianBlur'),
dict(p=0.5, type='HorizontalFlip'),
dict(p=0.5, type='VerticalFlip'),
dict(p=0.5, type='RandomRotate90'),
```

40k Iou

```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background |  95.4 | 98.81 |
|   garden   |  1.88 |  1.95 |
|    pond    |  8.84 | 11.73 |
|    soil    | 26.38 | 37.52 |
|  building  | 16.83 | 20.84 |
|  railway   |  0.0  |  0.0  |
|    road    |  5.27 |  5.86 |
+------------+-------+-------+
2024/03/30 20:02:27 - mmengine - INFO - Iter(val) [500/500]    aAcc: 95.2500  mIoU: 22.0800  mAcc: 25.2400  data_time: 0.0007  time: 0.0171
```
80k Iou
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 95.77 | 99.16 |
|   garden   | 11.51 | 15.71 |
|    pond    |  8.83 | 11.78 |
|    soil    | 25.55 | 30.78 |
|  building  | 18.51 | 22.28 |
|  railway   |  0.05 |  0.07 |
|    road    | 12.91 | 15.14 |
+------------+-------+-------+
2024/03/30 21:44:24 - mmengine - INFO - Iter(val) [500/500]    aAcc: 95.5900  mIoU: 24.7300  mAcc: 27.8500  data_time: 0.0007  time: 0.0172
```

# swin test2
config = ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512_match_noCrop_lowp.py

data_root ='E:\\lizhengcan\\ISPRS-CD\\train_set82'

pretrained = 'E:\\lizhengcan\\ISPRS-CD\\mmsegmentation-fix_cd_transform\\experiment\\swin-tiny-patch4-window7_upernet_1xb8-80k_levir-1024x1024\\iter_8000.pth'

增强
```
dict(type='RandomHistogramMatching'),
dict(p=0.2, type='CLAHE'),
dict(p=0.2, type='RandomBrightnessContrast'),
dict(p=0.2, type='RandomFog'),
dict(p=0.2, type='GaussianBlur'),
dict(p=0.5, type='HorizontalFlip'),
dict(p=0.5, type='VerticalFlip'),
dict(p=0.5, type='RandomRotate90'),

```
40k Iou
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 96.18 |  99.5 |
|   garden   |  4.88 |  5.29 |
|    pond    | 25.17 |  32.1 |
|    soil    | 29.88 | 35.35 |
|  building  | 17.71 |  19.5 |
|  railway   |  0.0  |  0.0  |
|    road    | 12.67 | 14.75 |
+------------+-------+-------+
2024/03/31 18:07:43 - mmengine - INFO - Iter(val) [500/500]    aAcc: 95.9900  mIoU: 26.6400  mAcc: 29.5000  data_time: 0.0008  time: 0.0174
```

80k Iou

```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background |  96.3 | 99.08 |
|   garden   | 17.94 |  25.5 |
|    pond    | 22.02 | 25.25 |
|    soil    | 33.94 | 42.28 |
|  building  | 22.77 | 32.44 |
|  railway   |  2.74 |  3.44 |
|    road    | 16.68 | 18.24 |
+------------+-------+-------+
2024/03/31 19:49:45 - mmengine - INFO - Iter(val) [500/500]    aAcc: 95.9700  mIoU: 30.3400  mAcc: 35.1800  data_time: 0.0008  time: 0.0174
```

# swin test3
pretrained = ALL-swin-tiny-patch4-window7_upernet_1xb8-80k_IS-512x512_noCrop_lowp.py

data_root ='E:\\lizhengcan\\ISPRS-CD\\train_set82'

pretrained = 'E:\\lizhengcan\\ISPRS-CD\\mmsegmentation-fix_cd_transform\\experiment\\swin-tiny-patch4-window7_upernet_1xb8-80k_levir-1024x1024\\iter_8000.pth'

增强
```
    dict(p=0.2, type='CLAHE'),
    dict(p=0.2, type='RandomBrightnessContrast'),
    dict(p=0.2, type='RandomFog'),
    dict(p=0.2, type='GaussianBlur'),
    dict(p=0.5, type='HorizontalFlip'),
    dict(p=0.5, type='VerticalFlip'),
    dict(p=0.5, type='RandomRotate90'),

```

40k Iou
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 95.73 | 99.31 |
|   garden   |  0.0  |  0.0  |
|    pond    |  3.89 |  3.95 |
|    soil    | 26.06 | 36.16 |
|  building  |  7.6  |  8.63 |
|  railway   |  0.0  |  0.0  |
|    road    |  0.08 |  0.08 |
+------------+-------+-------+
2024/03/31 23:50:43 - mmengine - INFO - Iter(val) [500/500]    aAcc: 95.4800  mIoU: 19.0500  mAcc: 21.1600  data_time: 0.0160  time: 0.0539
```

80k Iou
```
+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 95.99 | 98.77 |
|   garden   |  2.71 |  2.82 |
|    pond    | 23.09 | 30.91 |
|    soil    |  30.9 | 43.95 |
|  building  | 21.61 |  29.1 |
|  railway   |  2.72 |  2.88 |
|    road    | 23.83 | 28.81 |
+------------+-------+-------+
2024/04/01 04:21:37 - mmengine - INFO - Iter(val) [500/500]    aAcc: 95.6900  mIoU: 28.6900  mAcc: 33.8900  data_time: 0.0007  time: 0.0167
```