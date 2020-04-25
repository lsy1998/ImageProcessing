from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import math
 
# 最近邻插值算法
# dstH为新图的高;dstW为新图的宽
def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH-1):
        for j in range(dstW-1):
            scrx=round(i*(scrH/dstH))
            scry=round(j*(scrW/dstW))
            retimg[i,j]=img[scrx,scry]
    return retimg
 
im_path='../loginBackground.jpg'
image=np.array(Image.open(im_path))
# print(type(image.shape))
# height = image.shape[0]
image1=NN_interpolation(image, image.shape[0]*2, image.shape[1]*2)
image1=Image.fromarray(image1.astype('uint8')).convert('RGB')
image1.save('out.png')
