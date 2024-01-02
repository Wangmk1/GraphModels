import os
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm
from fire import Fire

r"""
    First, backlight removal:
        Backlight is a common problem in shooting. The backlight removal technology can effectively enhance the image 
        quality taken under backlight and significantly improve the accuracy of face recognition under backlight. 

    Second, low illumination enhancement:
        At night, insufficient illumination will lead to very poor image quality. Low illumination enhancement technology 
        can effectively enhance the brightness of the image and restore the details of the image, which is of great help 
        for video surveillance and license plate recognition at night. 

    Third, Deblur Image:
        blurring processing chooses to highlight or suppress some features in the image. By improving brightness, white 
        balance, noise removal, blur removal, fog removal and other functions, the image matches the visual response 
        characteristics, enhances the subjective effect, and makes the picture clearer and easier to watch.Blurring 
        phenomenon often appears in the imaging of moving objects, especially objects in high speed motion. Deblurring 
        technology can effectively enhance the clarity of blurred images, and it can play a very good auxiliary role for
        face recognition in motion and license plate recognition on the highway.

    Fourth, Haze removal: 
        The imaging of many images will be affected by the weather, especially the haze weather has a particularly 
        significant impact on the surveillance video.Image defogging technology can effectively restore the image and 
        video in haze state and improve the quality of surveillance video. 
    """


def check_report(image):
    assert torch.cuda.is_available(), ' Expected all tensors to be on the gpu device, please recheck!'
    assert type(image) == np.ndarray, ' Expected {} input but not {}, please recheck!'.format(np.ndarray, type(image))
    assert image.dtype == np.uint8, ' Expected {} input within the range of [0..255] but not {}, please recheck!'.format(
        np.uint8, image.dtype)


def back_lighting_compensation(image: "RGB numpy in [0..255]", level: int = 1) -> "float32 in [0..1]":
    check_report(image)

    tensor = torch.from_numpy(image).to('cuda')  # torch张量
    size = tensor.size()
    tensor = torch.reshape(tensor, [-1, 3])

    avg_RGB = torch.tensor([tensor[:, 0].sum(), tensor[:, 1].sum(), tensor[:, 2].sum()]).to('cuda') / tensor.size()[0]
    avg_Gray = avg_RGB.sum() / 3
    a_rgb = avg_Gray / avg_RGB
    a_rgb = torch.pow(a_rgb, level)

    tensor = tensor * a_rgb.T / 255
    tensor = torch.reshape(tensor, size)

    # 转到cpu-->生成numpy数组-->float32[0..1]
    return tensor.to('cpu').numpy()


def low_illumination_enhancement(image: "RGB numpy in [0..255]", level: int = 5) -> "float32 in [0..1]":
    check_report(image)

    kernel = np.array([[0, -1, 0], [0, level, 0], [0, -1, 0]], np.float32)  # 补光卷积核
    imageEnhance = cv2.filter2D(image, -1, kernel)

    return imageEnhance.astype(np.float32) / 255


def remove_indistinct(image: "RGB numpy in [0..255]", level: int = 7) -> "float32 in [0..1]":
    check_report(image)

    kernel = np.array([[-1, 1, -1], [-1, level, -1], [-1, -1, -1]], np.float32)  # 锐化卷积核
    imageEnhance = cv2.filter2D(image, -1, kernel)

    return imageEnhance.astype(np.float32) / 255


def zmMinFilterGray(src, r=7):
    """ 最小值滤波，r是滤波器半径 """
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):
    """计算大气遮罩图像V1和光照值A, V1 = 1-t/A"""
    V1 = np.min(m, 2)
    Dark_Channel = zmMinFilterGray(V1, 7)
    V1 = guidedfilter(V1, Dark_Channel, r, eps)
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for i in range(bins - 1, 0, -1):
        if d[i] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][i]].max()
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A


def remove_haze(image: "RGB numpy in [0..255]", r: int = 81, eps=0.001,
                w=0.95, maxV1=0.80, bGamma=False) -> "float64 in [0..1]":
    check_report(image)
    image = image / 255.0
    Y = np.zeros(image.shape)
    Mask_img, A = Defog(image, r, eps, w, maxV1)
    for k in range(3):
        Y[:, :, k] = (image[:, :, k] - Mask_img) / (1 - Mask_img / A)
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))
    return Y


def main(root_dir='images'):
    logging.basicConfig(datefmt="%Y-%M-%d %H:%M:%S", format="[%(asctime)s] %(levelname)s:%(message)s",
                        level=logging.INFO)
    image_number = len(os.listdir(root_dir))
    paths = sorted(os.listdir(root_dir))
    paths = map(os.path.join, [root_dir] * image_number, paths)
    images = map(cv2.imread, paths)
    images = list(map(cv2.resize, images, [(500, 330)] * image_number))

    """------------------Back Lighting Compensation-------------------------"""
    logging.info('\033[1;29mBack Lighting Compensation_>>>>>>>\033[0m')
    image_show1 = images[0].astype(np.float32) / 255  # unit8-->float32
    image_show2 = images[1].astype(np.float32) / 255  # unit8-->float32

    for i in tqdm(range(2, 5, 2)):
        image_i1 = back_lighting_compensation(images[0], i)  # 逆光补偿-->
        image_i2 = back_lighting_compensation(images[1], i)  # 逆光补偿-->
        image_show1 = np.append(image_show1, image_i1, axis=1)  # 同数据类-多图横向拼接-->
        image_show2 = np.append(image_show2, image_i2, axis=1)  # 同数据类-多图横向拼接-->
    image_show = np.append(image_show1, image_show2, axis=0)  # 同数据类-多图纵向拼接-->
    cv2.imshow('Back Lighting Compensation Level Up >>>>>>>', image_show)

    """----------------Low Illumination Enhancement-----------------------"""
    logging.info("\033[1;29mLow Illumination Enhancement_>>>>>>>\033[0m")
    image_show3 = images[2].astype(np.float32) / 255  # unit8-->float32
    image_show4 = images[3].astype(np.float32) / 255  # unit8-->float32

    for i in tqdm(range(2, 5, 2)):
        image_i1 = low_illumination_enhancement(images[2], 3 * i)  # 光照补偿-->
        image_i2 = low_illumination_enhancement(images[3], 2 * i)  # 光照补偿-->
        image_show3 = np.append(image_show3, image_i1, axis=1)  # 同数据类-多图横向拼接-->
        image_show4 = np.append(image_show4, image_i2, axis=1)  # 同数据类-多图横向拼接-->
    image_show = np.append(image_show3, image_show4, axis=0)  # 同数据类-多图纵向拼接-->
    cv2.imshow('Low Illumination Enhancement Level Up >>>>>>>', image_show)

    """----------------Remove Indistinct-----------------------"""
    logging.info("\033[1;29mRemove Indistinct_>>>>>>>\033[0m")
    image_show5 = images[4].astype(np.float32) / 255  # unit8-->float32
    image_show6 = images[5].astype(np.float32) / 255  # unit8-->float32

    for i in tqdm(range(2, 5, 2)):
        image_i1 = remove_indistinct(images[4])  # 去除模糊-->
        image_i2 = remove_indistinct(images[5])  # 去除模糊-->
        image_show5 = np.append(image_show5, image_i1, axis=1)  # 同数据类-多图横向拼接-->
        image_show6 = np.append(image_show6, image_i2, axis=1)  # 同数据类-多图横向拼接-->
    image_show = np.append(image_show5, image_show6, axis=0)  # 同数据类-多图纵向拼接-->
    cv2.imshow('Remove Indistinct Level Up >>>>>>>', image_show)

    """----------------Remove Haze-----------------------"""
    logging.info("\033[1;29mRemove Haze_>>>>>>>\033[0m")
    image_show7 = images[6].astype(np.float64) / 255.0  # unit8-->float32
    image_show8 = images[7].astype(np.float64) / 255.0  # unit8-->float32

    for i in tqdm(range(2, 5, 2)):
        image_i1 = remove_haze(images[6])  # 去除模糊-->
        image_i2 = remove_haze(images[7])  # 去除模糊-->
        image_show7 = np.append(image_show7, image_i1, axis=1)  # 同数据类-多图横向拼接-->
        image_show8 = np.append(image_show8, image_i2, axis=1)  # 同数据类-多图横向拼接-->
    image_show = np.append(image_show7, image_show8, axis=0)  # 同数据类-多图纵向拼接-->
    cv2.imshow('Remove Haze Level Up >>>>>>>', image_show)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)
