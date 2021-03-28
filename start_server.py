import argparse
import logging

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import socket
import threading
import time

import cv2

import network
from PIL import Image
import matplotlib.pyplot as plt
import utils
import os.path as osp
import os
import pickle
from pathlib import Path
from models.modules.Sakuya_arch import LunaTokis
from queue import Queue
import torch
import numpy as np
import models
from models.SR_model import SRModel
from models.SR_model_shared import SRModelShared
import schedular.datahandler
import schedular.threadHandler

def start_server_for_test():
    """
    用于从文件单独模拟服务端，由于可以模拟客户端弃用
    :return:
    """
    plt.ion()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 11111))
    server.listen(5)
    csock, _ = server.accept()
    plt.figure(1)
    cnt = 0
    data_dir = './test'
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    fs = open(osp.join(data_dir, ''), 'wb')
    while True:
        is_sample,  obj= network.handel_stream(csock)
        data=utils.decode_jpeg(['img'])
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
        pickle.dump(cache, fs, protocol=-1)
    fs.close()
    # if not is_sample:
    #     plt.imshow(data)
    #     plt.draw()
    #     plt.show()
    # else:
    #     plt.imshow(utils.decode_jpeg(data['LR']))
    #     plt.draw()
    #     plt.show()
    #     plt.imshow(utils.decode_jpeg(data['HR']))
    #     plt.draw()
    #     plt.show()
def get_weight(self,img):
    """
    :param img:输入的图像tensor，结构为(H,W,RGB)
    :return: 返回拉普拉斯卷积的结果，目前认为该值可以衡量细节的多少，用于筛选细节较多的样本，但事实上有可能并不够理想
            可能需要更好的指标
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = cv2.Laplacian(gray, cv2.CV_64F).var()
    return result

def test_pkl(file_path):
    logger=logging.getLogger('base')
    model_path='./test/saved_model.pth'
    fpath=Path(file_path)
    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    if not fpath.is_file():
        logger.info('no such file!')
        exit(0)
    plt.ion()
    fs=open(file_path,'rb')
    flag=True
    sr_model=SRModel(model_path,learning_rate=1e-4)#此处的学习率目前看来可以接受，但是不能完全确定
    prev=None
    samples_lq=[]
    samples_gt=[]
    sample_cnt=0
    cnt=0
    while flag:
        try:
            data=pickle.load(fs)
            is_sample=data['is_sample']
            data=data['data']
            if not is_sample:
                logger.info(data.shape)
                # plt.imshow(data)
                # plt.draw()
                # plt.show()
                resized=cv2.resize(data,(data.shape[1]*4,data.shape[0]*4),cv2.INTER_NEAREST)/255.0
                cnt = cnt + 1
                if cnt%10==0:
                    plt.imshow(resized)
                    plt.draw()
                    plt.show()
                data=data.transpose((2, 0, 1))/255.#because the original data is 0-255,should
                                                    #be scaled to 0-1
                if prev is None:
                    prev=data
                    data=np.stack([data,data],axis=0)
                else:
                    tmp=data
                    data=np.stack([prev,data],axis=0)
                    prev=tmp
                data=torch.from_numpy(np.ascontiguousarray(data)).float()
                data=data.unsqueeze(0).to(device)
                logger.info(data.shape)
                sr_model.pred_model.eval()
                with torch.no_grad():
                    output=sr_model.forward(data)
                if isinstance(output,list) or isinstance(output,tuple):
                    output=output[0]
                output=output.data.float().cpu()
                logger.info('output shape:',output.shape)
                if cnt% 10==0:
                    img=output[0,0,:,:,:].squeeze(0).squeeze(0).numpy()
                    img=img.transpose((1,2,0))
                    logger.info(img.shape)
                    plt.imshow(img)
                    plt.draw()
                    plt.show()
                # for i in range(1):
                #     img=output[0,i,:,:,:].squeeze(0).squeeze(0).numpy()
                #     img=img.transpose((1,2,0))
                #     print(img.shape)
                #     plt.imshow(img)
                #     plt.draw()
                #     plt.show()
            else:
                data['LQs']=[utils.decode_jpeg(lr)for lr in data['LQs']]
                data['GT']=[utils.decode_jpeg(hr)for hr in data['GT']]
                data['LQs']=np.stack(data['LQs'],axis=0)
                data['GT']=np.stack(data['GT'],axis=0)
                data['LQs']=torch.from_numpy(np.ascontiguousarray(np.transpose(data['LQs'],(0,3,1,2)))).float()
                data['GT'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['GT'], (0, 3, 1, 2)))).float()
                data['key']=time.time()
                sample_cnt=sample_cnt+1
                samples_lq.append(data['LQs'])
                samples_gt.append(data['GT'])
                if sample_cnt ==5:
                    sample_cnt=0
                    data['GT']=torch.stack(samples_gt,dim=0)/255.#to0-1
                    data['LQs']=torch.stack(samples_lq,dim=0)/255.
                    sr_model.learning(data)
                    samples_lq.clear()
                    samples_gt.clear()
                # plt.imshow(utils.decode_jpeg(data['LR']))
                # plt.draw()
                # plt.show()
                # plt.imshow(utils.decode_jpeg(data['HR']))
                # plt.draw()
                # plt.show()
        except EOFError:
            logger.info('INFO:end of file')
            flag=False
    fs.close()

def test_PSNR(model_path,file_path):
    plt.ion()
    logger = logging.getLogger('base')
    sr_model = SRModel(model_path, learning_rate=5e-4)  # 此处的学习率目前看来可以接受，但是不能完全确定
    fs = open(file_path, 'rb')
    flag = True
    prev = None
    hr_list=[]
    samples_lq = []
    samples_gt = []
    sample_cnt = 0
    cnt = 0
    batch_size=5
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
                if len(hr_list)==0:
                    hr_list.append(utils.decode_jpeg(data['origin']).transpose((2, 0, 1)) / 255.)
                    prev=utils.decode_jpeg(data['scaled']).transpose((2, 0, 1)) / 255.
                elif len(hr_list)==1:
                    hr_list.append(utils.decode_jpeg(data['origin']).transpose((2, 0, 1)) / 255.)
                else:
                    hr_list.append(utils.decode_jpeg(data['origin']).transpose((2, 0, 1)) / 255.)
                    tmp=utils.decode_jpeg(data['scaled']).transpose((2, 0, 1)) / 255.
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
                    show=img1[0,:,:,:].transpose(1,2,0)/255.
                    #show=show[:,:,::-1]
                    plt.imshow(show)
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
def start_server_shared(model_path,ip_addr,port):
    plt.ion()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip_addr, port))
    server.listen(5)
    logger=logging.getLogger('base')
    sr_model = SRModelShared(model_path, learning_rate=1e-4)  # 此处的学习率目前看来可以接受，但是不能完全确定
    thread_list=[]
    while True:
        csock,addr=server.accept()
        logger.info('incomming connection,ip:{} port:{}'.format(addr[0],addr[1]))
        thread=schedular.datahandler.ClientHandler(csock,sr_model)
        thread_list.append(thread)
        thread.setDaemon(True)
        thread.start()
        #TODO  添加schedular对线程进行处理
        for i in range(len(thread_list)-1,-1,-1):#倒序遍历
            if thread_list[i].is_terminated:
                thread_list.pop(i)

def start_server(model_path,ip_addr,port):
    plt.ion()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip_addr, port))
    server.listen(5)
    logger=logging.getLogger('base')

    thread_list=[]

    #use thread handler to manage threads
    thread_handler=schedular.threadHandler.threaed_handler(thread_list,logger)
    thread_handler.setDaemon(True)
    thread_handler.start()
    while True:
        csock,addr=server.accept()
        logger.info('incomming connection,ip:{} port:{}'.format(addr[0],addr[1]))
        sr_model = SRModel(model_path, learning_rate=1e-4)  # 此处的学习率目前看来可以接受，但是不能完全确定
        thread=schedular.datahandler.ClientHandler(csock,sr_model)
        thread_list.append(thread)
        thread.setDaemon(True)
        thread.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='individual',
                        help='work mode select,expect \'shared\' or \'individual\' or \'psnr\',default \'individual\'')
    args = parser.parse_args()
    utils.setup_logger('base',False)#default without logfile
    if args.mode=='psnr':
        test_PSNR('./test/10000_G.pth','./test/psnr_data.pkl')
    elif args.mode=='individual':
        start_server('./test/saved_model.pth','0.0.0.0',11111)
    elif args.mode=='shared':
        start_server_shared('./test/saved_model.pth','0.0.0.0',11111)
    else :
        print('mode should be \'individual\' or \'shared\'')
        exit(-1)
    #test_pkl('./test/cahce_new.pkl')