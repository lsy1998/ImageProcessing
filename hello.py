from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from math import *
import cv2
import time
from flask import *
# Flask, redirect, url_for, request, ke_response, jsonify, Response
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources=r'/*')

# 最近邻插值
@app.route('/jiaocheng', methods=['POST '])
def success():
    # user = request.form['name']
    print(request.get_json())
    params = request.get_json()
    # print(type(params))
    name = params['name']
    num1 = int(params['num1'])
    num2 = int(params['num2'])
    print(num1, num2)

    def NN_interpolation(img, dstH, dstW):
        scrH, scrW, s = img.shape
        retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
        for i in range(dstH-1):
            for j in range(dstW-1):
                scrx = round(i*(scrH/dstH))
                scry = round(j*(scrW/dstW))
                retimg[i, j] = img[scrx, scry]
        return retimg

    im_path = f'./{name}'
    image = np.array(Image.open(im_path))
# print(type(image.shape))
# height = image.shape[0]
    image1 = NN_interpolation(image, image.shape[0]*num1, image.shape[1]*num2)
    image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
    image1.save(r'C:\Users\hasee\Desktop\jiaocheng\out3.png')
    # response = make_response(jsonify({'name': 'out2.pang'}, 200)
    t = {
        'name': 'out3.png',
    }
    return json.dumps(t)

# 双线性插值
@app.route('/shuangxianxing', methods=['POST'])
def shuangxianxing():
    # user = request.form['name']
    # print(request.get_json())
    # params = request.get_json()
    # # print(type(params))
    # name = params['name']
    # num1 = int(params['num1'])
    # num2 = int(params['num2'])
    # print(num1,num2)
    def bilinear_interpolation(img, out_dim):
        src_h, src_w, channel = img.shape
        dst_h, dst_w = out_dim[1], out_dim[0]
        if src_h == dst_h and src_w == dst_w:
            return img.copy()
        dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
        scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
        for i in range(channel):
            for dst_y in range(dst_h):
                for dst_x in range(dst_w):
                    src_x = (dst_x + 0.5) * scale_x - 0.5
                    src_y = (dst_y + 0.5) * scale_y - 0.5
                    src_x0 = int(floor(src_x))
                    src_x1 = min(src_x0 + 1, src_w - 1)
                    src_y0 = int(floor(src_y))
                    src_y1 = min(src_y0 + 1, src_h - 1)
                    if src_x0 != src_x1 and src_y1 != src_y0:
                        temp0 = ((src_x1 - src_x) * img[src_y0, src_x0, i] + (
                            src_x - src_x0) * img[src_y0, src_x1, i]) / (src_x1 - src_x0)
                        temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (
                            src_x - src_x0) * img[src_y1, src_x1, i] / (src_x1 - src_x0)
                        dst_img[dst_y, dst_x, i] = int(
                            (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1) / (src_y1 - src_y0)
        return dst_img
    im_path = './loginBackground.jpg'
    img = np.array(Image.open(im_path))
    # img = cv2.imread('./loginBackground.jpg')
    start = time.time()
    image1 = bilinear_interpolation(img, (100, 100))
    print('cost {} seconds'.format(time.time() - start))
    # cv2.imshow('result', dst)
    # cv2.waitKey()
#     im_path = f'./{name}'
#     image = np.array(Image.open(im_path))
# # print(type(image.shape))
# # height = image.shape[0]
#     image1 = NN_interpolation(image, image.shape[0]*num1, image.shape[1]*num2)
    image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
    image1.save(r'C:\Users\hasee\Desktop\jiaocheng\out4.png')
    # response = make_response(jsonify({'name': 'out2.pang'}, 200)
    t = {
        'name': 'out4.png',
    }
    return json.dumps(t)


#边缘提取
@app.route('/gaotong', methods=['POST'])
def gaotong():
    # user = request.form['name']
    # print(request.get_json())
    # params = request.get_json()
    # # print(type(params))
    # name = params['name']
    # num1 = int(params['num1'])
    # num2 = int(params['num2'])
    # print(num1,num2)
    kernel_9 = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])


    kernel_25 = np.array([[-1, -1, -1, -1, -1],
                      [-1, 1, 2, 1, -1],
                      [-1, 2, 4, 2, -1],
                      [-1, 1, 2, 1, -1],
                      [-1, -1, -1, -1, -1]])

    img = cv2.imread('./loginBackground.jpg')
    ndimg = np.array(img)

    # k3 = cv2.filter2D(ndimg, -1, kernel_9)  # convolve calculate
    # the second parameters measns the deepth of passageway.
    k5 = cv2.filter2D(ndimg, -1, kernel_25)
    src = cv2.imread("./loginBackground.jpg")
    # htich = np.hstack((src, k3))
    cv2.imwrite(r'C:\Users\hasee\Desktop\jiaocheng\out5_ps.png', k5)
    # such as cv2.CV_8U means every passageway is 8 bit.
    # -1 means the passageway of the source file and the object file is equal.
    # plt.subplot(131)
    # plt.imshow(img)
    # plt.title("source image")

    # plt.subplot(132)
    # plt.imshow(k3)
    # plt.title("kernel = 3")
    # x = np.arange(5)
    # y = x
    # plt.plot(x, y, '-o')
    # plt.imshow(k5)
    # plt.title("kernel = 5")
    # plt.savefig(r'C:\Users\hasee\Desktop\jiaocheng\out5_ps.png')
    # plt.show()

# response = make_response(jsonify({'name': 'out2.pang'}, 200)
    t = {
        'name': 'out4.png',
    }
    return json.dumps(t)

#canny边缘检测
@app.route('/canny', methods=['POST'])
def canny():
    img = cv2.imread('./loginBackground.jpg')
    data = (100, 300)
    # cv2.imshow('img-Canny', cv2.Canny(img, *data))
    cv2.imwrite(r'C:\Users\hasee\Desktop\jiaocheng\Canny.jpg', cv2.Canny(img, *data))
    t = {
        'name': 'out4.png',
    }
    return json.dumps(t)

#拉普拉斯边缘检测
@app.route('/Laplacian', methods=['POST'])
def Laplacian():
    image = cv2.imread("./loginBackground.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
    # cv2.imshow("Original",image)
    # cv2.waitKey()

    #拉普拉斯边缘检测
    lap = cv2.Laplacian(image,cv2.CV_64F)#拉普拉斯边缘检测
    lap = np.uint8(np.absolute(lap))##对lap去绝对值
    cv2.imwrite(r'C:\Users\hasee\Desktop\jiaocheng\Laplacian.jpg', lap)
    # cv2.imshow("Laplacian",lap)
    t = {
        'name': 'out4.png',
    }
    return json.dumps(t)

#Soble边缘检测
@app.route('/Soble', methods=['POST'])
def Soble():
    image = cv2.imread("./loginBackground.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
    cv2.imshow("Original",image)
    cv2.waitKey()

    #Sobel边缘检测
    sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)#x方向的梯度
    sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)#y方向的梯度

    sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值

    sobelCombined = cv2.bitwise_or(sobelX,sobelY)#
    cv2.imshow("Sobel X", sobelX)
    cv2.waitKey()
    cv2.imshow("Sobel Y", sobelY)
    cv2.waitKey()
    cv2.imshow("Sobel Combined", sobelCombined)
    cv2.waitKey()
    t = {
        'name': 'out4.png',
    }
    return json.dumps(t)

#大津法阈值分割
@app.route('/dajin', methods=['POST'])
def dajin():
    def rgb2gray(img):
        h=img.shape[0]
        w=img.shape[1]
        img1=np.zeros((h,w),np.uint8)
        for i in range(h):
            for j in range(w):
                img1[i,j]=0.144*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,1]
        return img1

    def otsu(img):
        h=img.shape[0]
        w=img.shape[1]
        m=h*w   # 图像像素点总和
        otsuimg=np.zeros((h,w),np.uint8)
        threshold_max=threshold=0   # 定义临时阈值和最终阈值
        histogram=np.zeros(256,np.int32)   # 初始化各灰度级个数统计参数
        probability=np.zeros(256,np.float32)   # 初始化各灰度级占图像中的分布的统计参数
        for i in range (h):
            for j in range (w):
                s=img[i,j]
                histogram[s]+=1   # 统计像素中每个灰度级在整幅图像中的个数
        for k in range (256):
            probability[k]=histogram[k]/m   # 统计每个灰度级个数占图像中的比例
        for i in range (255):
            w0 = w1 = 0   # 定义前景像素点和背景像素点灰度级占图像中的分布
            fgs = bgs = 0   # 定义前景像素点灰度级总和and背景像素点灰度级总和
            for j in range (256):
                if j<=i:   # 当前i为分割阈值
                    w0+=probability[j]   # 前景像素点占整幅图像的比例累加
                    fgs+=j*probability[j]
                else:
                    w1+=probability[j]   # 背景像素点占整幅图像的比例累加
                    bgs+=j*probability[j]
            u0=fgs/w0   # 前景像素点的平均灰度
            u1=bgs/w1   # 背景像素点的平均灰度
            g=w0*w1*(u0-u1)**2   # 类间方差
            if g>=threshold_max:
                threshold_max=g
                threshold=i
        print(threshold)
        for i in range (h):
            for j in range (w):
                if img[i,j]>threshold:
                    otsuimg[i,j]=255
                else:
                    otsuimg[i,j]=0
        return otsuimg

    image = cv2.imread("./loginBackground.jpg")
    grayimage = rgb2gray(image)
    otsuimage = otsu(grayimage)
    cv2.imshow("image", image)
    cv2.imshow("grayimage",grayimage)
    cv2.imshow("otsu", otsuimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    t = {
        'name': 'out4.png',
    }
    return json.dumps(t)


@app.route('/fuliye', methods=['POST'])
def fuliye():
    def dft(img):
        print(111)
        H, W, channel = img.shape
        # Prepare DFT coefficient
        G = np.zeros((H, W, channel), dtype=np.complex)
        print(G)
        # prepare processed index corresponding to original image positions
        x = np.tile(np.arange(W), (H, 1))
        y = np.arange(H).repeat(W).reshape(H, -1)
        print(channel, H, W)
        # dft
        for c in range(channel):
            for v in range(H):
                for u in range(W):
                    print(u)
                    G[v, u, c] = np.sum(
                        img[..., c] * np.exp(-2j * np.pi * (x * u / W + y * v / H))) / np.sqrt(H * W)
        print(G)
        return G
# IDFT

    def idft(G):
        # prepare out image
        H, W, channel = G.shape
        out = np.zeros((H, W, channel), dtype=np.float32)
        # prepare processed index corresponding to original image positions
        x = np.tile(np.arange(W), (H, 1))
        y = np.arange(H).repeat(W).reshape(H, -1)
        # idft
        for c in range(channel):
            for v in range(H):
                for u in range(W):
                    out[v, u, c] = np.abs(np.sum(
                        G[..., c] * np.exp(2j * np.pi * (x * u / W + y * v / H)))) / np.sqrt(W * H)
        # clipping
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        return out

    # Read image
    img = cv2.imread("./loginBackground.jpg").astype(np.float32)
    # DFT
    G = dft(img)
    print(G)
    # write poser spectal to image
    ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
    cv2.imwrite(r'C:\Users\hasee\Desktop\jiaocheng\out5_ps.png', ps)
    # IDFT
    out = idft(G)
    # Save result
    cv2.imshow("result", out)
    cv2.imwrite(r'C:\Users\hasee\Desktop\jiaocheng\out5.png', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    t = {
        'name': 'out5.png',
    }
    return json.dumps(t)


if __name__ == '__main__':
    app.run(debug=True)
