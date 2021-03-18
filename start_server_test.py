import logging
import pickle
import socket
import time

import cv2
import matplotlib.pyplot as plt
import os.path as osp
import os

import torch
import numpy as np
from numpy import sort

import network


import utils
from models.SR_model_shared import SRModelShared


def test_pkl(file_path):
    plt.ion()
    fs=open(file_path,'rb')
    flag=True
    prev=None
    samples_lq=[]
    samples_gt=[]
    sample_cnt=0
    cnt=0
    while flag:
        try:
            data=pickle.load(fs)
            origin=data.get('origin')
            if origin is None:
                pass
                #print(data)
            else:
                print(data)
                plt.imshow(utils.decode_jpeg(origin))
                plt.draw()
                plt.show()
                print('image')
        except EOFError:
            flag=False
    fs.close()

def start_server():
    plt.ion()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 11111))
    server.listen(5)
    csock, _ = server.accept()
    plt.figure(1)
    cnt = 0
    # data_dir = './test'
    # if not osp.exists(data_dir):
    #     os.makedirs(data_dir)
    #fs = open(osp.join(data_dir, 'cahce_new.pkl'), 'wb')
    while True:
        is_sample, obj = network.handel_stream(csock)
        data=utils.decode_jpeg(obj['img'])
        # if not data:
        #     print('trans end')
        #     break
        print(is_sample)
        if data is None:
            break
        print(data)
        if not is_sample:
            print(data.shape)
        cache = {'is_sample': is_sample, 'data': data}
        #pickle.dump(cache, fs, protocol=-1)
        if not is_sample:
            pass
            print('get data'+str(data.shape))
            obj={'loss':500.0,'delta':-0.01}
            network.send_configuration(csock,obj)
            # plt.imshow(data)
            # plt.draw()
            # plt.show()
        else:
            pass
            print('get sample')
            # for img in data['LQs']:
            #     plt.imshow(utils.decode_jpeg(img))
            #     plt.draw()
            #     plt.show()
            # for img in data['GT']:
            #     plt.imshow(utils.decode_jpeg(img))
            #     plt.draw()
            #     plt.show()
    #fs.close()

def test_PSNR(model_path,file_path,lr_path):
    plt.ion()
    logger = logging.getLogger('base')
    sr_model = SRModelShared(model_path, learning_rate=5e-4)  # 此处的学习率目前看来可以接受，但是不能完全确定
    fs = open(file_path, 'rb')
    flag = True
    prev = None
    hr_list=[]
    samples_lq = []
    samples_gt = []
    sample_cnt = 0
    cnt = 0
    batch_size=5
    lr_list=os.listdir(lr_path)
    lr_list=sort(lr_list)
    while flag:
        try:
            data = pickle.load(fs)
            origin = data.get('origin')
            if origin is None:
                data['LQs'] = [utils.decode_jpeg(lr) for lr in data['LQs']]
                data['GT'] = [utils.decode_jpeg(hr) for hr in data['GT']]
                data['LQs'] = np.stack(data['LQs'], axis=0)
                data['GT'] = np.stack(data['GT'], axis=0)
                data['LQs'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['LQs'], (0, 3, 1, 2)))).float()
                data['GT'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['GT'], (0, 3, 1, 2)))).float()
                data['key'] = time.time()
                # 准备数据
                sample_cnt = sample_cnt + 1
                samples_lq.append(data['LQs'])
                samples_gt.append(data['GT'])
                if sample_cnt == batch_size:  # 样本达到一定数量后开始训练
                    sample_cnt = 0
                    data['GT'] = torch.stack(samples_gt, dim=0) / 255.  # to0-1
                    data['LQs'] = torch.stack(samples_lq, dim=0) / 255.
                    sr_model.learning(data)
                    samples_lq.clear()
                    samples_gt.clear()
            else:
                img_path=osp.join(lr_path,lr_list[cnt])
                cnt=cnt+1
                if len(hr_list)==0:
                    hr_list.append(utils.decode_jpeg(data['origin']).transpose((2, 0, 1)) / 255.)
                    prev=cv2.imread(img_path,cv2.IMREAD_COLOR)
                    prev=np.copy(prev[:,:,::-1])
                    prev=cv2.resize(prev,(160,120),interpolation=cv2.INTER_NEAREST).transpose((2, 0, 1)) / 255.
                    #prev=utils.decode_jpeg(data['scaled']).transpose((2, 0, 1)) / 255.
                elif len(hr_list)==1:
                    hr_list.append(utils.decode_jpeg(data['origin']).transpose((2, 0, 1)) / 255.)
                else:
                    hr_list.append(utils.decode_jpeg(data['origin']).transpose((2, 0, 1)) / 255.)
                    tmp = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    tmp = np.copy(tmp[:, :, ::-1])
                    tmp=cv2.resize(tmp,(160,120),interpolation=cv2.INTER_NEAREST).transpose((2, 0, 1)) / 255.
                    #tmp=utils.decode_jpeg(data['scaled']).transpose((2, 0, 1)) / 255.
                    data = np.stack([prev, tmp], axis=0)
                    data = torch.from_numpy(np.ascontiguousarray(data)).float()
                    data = data.unsqueeze(0)
                    sr_model.pred_model.eval()
                    with torch.no_grad():
                        output = sr_model.forward(data)
                    if isinstance(output, list) or isinstance(output, tuple):
                        output = output[0]
                    output = output.data.float().cpu()
                    logger.info(output.shape)
                    img1=np.array(output)*255
                    img1=img1.squeeze(0)
                    img2=np.array(hr_list)*255
                    plt.imshow(img1[0,:,:,:].transpose(1,2,0)/255.)
                    plt.draw()
                    plt.show()
                    #logger.info(img1[0,:,:,:].shape)
                    #logger.info(img2.shape)
                    cal_psnr=utils.calculate_psnr(img1,img2)
                    #cal_ssim=utils.calculate_ssim(img1,img2)
                    logger.info('\nPSNR:{}  SSIM:N/A\n'.format(cal_psnr))
                    hr_list.clear()
        except EOFError:
            flag = False
    fs.close()

if __name__ == '__main__':
    utils.setup_logger('base',True)
    #start_server()
    # utils.create_SR_resized('/home/wsn/LRC_div/Datasets/COCO/val2017/'
    #                         ,'/home/wsn/LRC_div/Datasets/COCO/SR2017/'
    #                         ,'/home/wsn/LRC_div/Datasets/COCO/resize2017/')
    utils.create_SR_resized('/mnt/data/LRC_DATA/LRC_div/Datasets/semantic-segmentation/data/ADEChallengeData2016/images/training'
                            ,'/mnt/data/LRC_DATA/LRC_div/Datasets/semantic-segmentation/data/ADEChallengeData2016/images/training_SR'
                            ,'/mnt/data/LRC_DATA/LRC_div/Datasets/semantic-segmentation/data/ADEChallengeData2016/images/training_RE')
    #test_PSNR('./test/saved_model.pth','/mnt/data/BaiduNetDiskDownload/Vid4_pkl/calendar.pkl','/mnt/data/BaiduNetDiskDownload/Vid4/LR/calendar/')