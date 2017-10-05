# -*- coding: utf-8 -*-
'''
    python图片处理
'''
from PIL import Image
from skimage import io
import numpy as np
from sklearn import preprocessing

def resizeImg(srcImgFile, dstImgFile, width=28, height=28):
    img = Image.open(srcImgFile)
    new_img = img.resize((width, height), Image.BILINEAR)
    new_img.save(dstImgFile)

def getImgAsMatFromFile(filename, width=28, height=28, scale_min=0, scale_max=1):
    #img = io.imread(filename, as_grey=True)
    img = Image.open(filename)
    img = img.resize((width, height), Image.BILINEAR)
    imgArr_2d = np.array(img.convert('L'))
    imgArr_2d = np.float64(1 - imgArr_2d)
    shape_2d = imgArr_2d.shape
    imgArr_1d_scale = preprocessing.minmax_scale(imgArr_2d.flatten(), feature_range=(0, 1))
    return imgArr_1d_scale.reshape(shape_2d)

def getImgMat(srcImgFile, width=28, height=28):
    img = Image.open(srcImgFile)
    new_img = img.resize((width, height), Image.BILINEAR)
    return np.array(new_img)


filename = 'test_data/9.jpg'
#resizeImg('test_data/8.jpg', 'test_data/8_28x28.jpg', 28, 28)
imgMat = getImgAsMatFromFile(filename)

