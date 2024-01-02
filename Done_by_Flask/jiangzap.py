import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from PIL.Image import Image
from numpy import ubyte
from torch import uint8

o = cv2.imread("1.png", 0)

# 直方图均衡化处理（只能处理灰度图片）
# clipLimit颜色对比度的阈值， titleGridSize进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
cl1 = clahe.apply(o)

# 闭与开相反
he = np.ones((2, 2), np.uint8)
bi = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, he)

# 均值滤波
jun = cv2.blur(bi, (2, 2))

# # 双边滤波
blur = cv2.bilateralFilter(jun, 0, 90, -100)

# 进行傅里叶变换
dft = cv2.dft(np.float32(blur), flags=cv2.DFT_COMPLEX_OUTPUT)
# 进行低频位置移动到中心
dshift = np.fft.fftshift(dft)
# 找到行列的值
rs, cs = blur.shape
# 计算中心的行和列
cr, cc = int(rs / 2), int(cs / 2)

mask = np.zeros((rs, cs, 2), np.uint8)
for i in range(rs):
    for j in range(cs):
        mask[i][j] = 1

m = -20  # 越小越清晰，越大越糊
n = 185  # 越小越糊，越大越清晰
mask[int(rs / 2 - m):int(rs / 2 + m), int(cs / 2 - m):int(cs / 2 + m)] = 0  # 高通
mask[int(rs / 2 - n):int(rs / 2 + n), int(cs / 2 - n):int(cs / 2 + n)] = 1  # 低通
md = dshift * mask
# 逆傅里叶变换
imd = np.fft.ifftshift(md)
io = cv2.idft(imd)
io = cv2.magnitude(io[:, :, 0], io[:, :, 1])

# 图像归一化处理
io = (io - np.min(io)) / (np.max(io) - np.min(io))

# 反归一化处理
io = io * 255
print("io", type(io))
# 整型转换
io = np.uint8(io)
print(io)
clahe1 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6, 6))
cl2 = clahe1.apply(io)
cv2.imshow('io', cl2)
cv2.waitKey(0)
cv2.destroyAllWindows()
