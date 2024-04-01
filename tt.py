from skimage import data
from skimage.io import imread, imsave
from skimage.exposure import match_histograms

# # 读取底图和要匹配的图像
# base_img = cv2.imread(r'E:\lizhengcan\ISPRS-CD\train_set82\test\T1\2238第二期影像.tif')
# target_img = cv2.imread(r'E:\lizhengcan\ISPRS-CD\train_set82\test\T2\2238第二期影像.tif')
# 读取两个图像
reference = imread(r'E:\lizhengcan\ISPRS-CD\train_set82\test\T1\2238第二期影像.tif')
image = imread(r'E:\lizhengcan\ISPRS-CD\train_set82\test\T2\2238第三期影像.tif')

# 直方图匹配
matched = match_histograms(image, reference, channel_axis=2)

# 保存结果
imsave('matched.png', matched)