
import logging
import math
import os
import os.path as osp
import random
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
#import torch
import cv2

from models.SR_model import SRModel
import torch


def decode_jpeg(img_str):
    data = np.frombuffer(img_str, np.uint8)  # 将获取到的字符流数据转换成1维数组
    #print(img_str)
    decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
    return decimg

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, tofile=False,file_name='test',root='./logs',screen=True, level=logging.INFO):
    '''设定logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, '{}_{}.log'.format(file_name,get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1)
    # print('img1-2')
    # print(img2)
    mse = np.mean((img1 - img2)**2)
    # print(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def create_SR_resized(in_dir,SR_dir,resized_dir):
    plt.ion()
    if not osp.exists(in_dir):
        print('no such input dir!')
        exit(-1)
    if not osp.exists(SR_dir):
        os.makedirs(SR_dir)
    if not osp.exists(resized_dir):
        os.makedirs(resized_dir)
    sr_model = SRModel('../test/saved_model.pth', learning_rate=1e-4)  # 此处的学习率目前看来可以接受，但是不能完全确定
    sr_model.pred_model.eval()
    file_list=os.listdir(in_dir)
    cnt=0
    for fname in file_list:
        cnt=cnt+1
        #print(osp.join(in_dir,fname))
        img_raw=cv2.imread(osp.join(in_dir,fname))
        # plt.imshow(img_raw)
        # plt.draw()
        # plt.show()
        shape_raw=img_raw.shape
        #print((4*np.ceil(shape_raw[0]/4),4*np.ceil(shape_raw[1]/4),shape_raw[2]))
        new_y=int(16*np.ceil(shape_raw[0]/16))
        new_x=int(16*np.ceil(shape_raw[1]/16))
        #print(new_x,new_y)
        tmp=np.zeros((new_y,new_x,shape_raw[2]),dtype=np.float32)
        tmp[0:shape_raw[0],0:shape_raw[1],:]=img_raw #make it scalable
        img_raw=None
        scaled=cv2.resize(tmp,dsize=(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
        resized=cv2.resize(scaled,dsize=(0,0),fx=4,fy=4,interpolation=cv2.INTER_AREA)[0:shape_raw[0],0:shape_raw[1],:]
        cv2.imwrite(osp.join(resized_dir,fname),resized)
        resized=None

        scaled=scaled.transpose((2, 0, 1)) /255.
        scaled=scaled[::-1,:,:] #convert to RGB
        data=np.stack([scaled, scaled], axis=0)
        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        data = data.unsqueeze(0)
        #print(data.shape)
        with torch.no_grad():
            torch.cuda.empty_cache()
            output = sr_model.forward(data)
            torch.cuda.empty_cache()
        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0]
        output = output.data.float().cpu()
        tmp = output[0, 0, :, :, :].squeeze(0).squeeze(0).numpy()
        tmp=tmp[::-1,:,:]
        tmp = tmp.transpose((1, 2, 0))[0:shape_raw[0], 0:shape_raw[1], :]  # 切取相同的大小
        # plt.imshow(tmp)
        # plt.draw()
        # plt.show()
        cv2.imwrite(osp.join(SR_dir,fname),tmp*255)
        if cnt%10==0:
            print('complete:{}/{}'.format(cnt,len(file_list)))