import logging
import threading
import time
import matplotlib.pyplot as plt

import cv2
import torch

import network
import utils
import numpy as np
from models.SR_model import SRModel

class ClientHandler(threading.Thread):
    #TODO 添加线程对外开放的变量，用来进行调度处理，如模型表现等等
    def __init__(self, csock,sr_model,batch_size=3):
        super().__init__()
        self.client_sock=csock
        self.sr_model=sr_model
        self.terminate=False    #主动停止，用于控制
        self.batch_size=batch_size
        self.logger=logging.getLogger('base')
        self.is_terminated=False#状态变量，用于判断是否结束
    def run(self) -> None:
        plt.ion()
        samples_lq=[]
        samples_gt=[]
        sample_cnt=0
        prev=None
        while not self.terminate:
            is_sample,obj=network.handel_stream(self.client_sock)
            if not is_sample and obj is None:
                self.terminate=True
                break
            if not is_sample:
                self.logger.info(obj['shape'])
                # plt.imshow(data)
                # plt.draw()
                # plt.show()
                #resized = cv2.resize(data, (data.shape[1] * 4, data.shape[0] * 4), cv2.INTER_NEAREST) / 255.0
                data = utils.decode_jpeg(obj['img'])
                data = data.transpose((2, 0, 1)) /255. # because the original data is 0-255,should
                # be scaled to 0-1
                if prev is None:
                    prev = data
                    continue
                else:
                    tmp = data
                    data = np.stack([prev, data], axis=0)
                    prev = tmp
                data = torch.from_numpy(np.ascontiguousarray(data)).float()
                data = data.unsqueeze(0)
                self.logger.info(data.shape)
                self.sr_model.pred_model.eval()
                with torch.no_grad():
                    output = self.sr_model.forward(data)
                if isinstance(output, list) or isinstance(output, tuple):
                    output = output[0]
                output = output.data.float().cpu()
                #TODO 处理output
                self.logger.info('output shape:'+str(output.shape))
                for i in range(2):
                    tmp=output[0,i+1,:,:,:].squeeze(0).squeeze(0).numpy()
                    tmp=tmp.transpose((1,2,0))[0:obj['shape'][1],0:obj['shape'][0],:]#切取相同的大小
                    plt.imshow(tmp)
                    plt.draw()
                    plt.show()
                # if cnt % 10 == 0:
                #     img = output[0, 0, :, :, :].squeeze(0).squeeze(0).numpy()
                #     img = img.transpose((1, 2, 0))
                #     logger.info(img.shape)
                #     plt.imshow(img)
                #     plt.draw()
                #     plt.show()
                # for i in range(1):
                #     img=output[0,i,:,:,:].squeeze(0).squeeze(0).numpy()
                #     img=img.transpose((1,2,0))
                #     print(img.shape)
                #     plt.imshow(img)
                #     plt.draw()
                #     plt.show()
            else:
                data=obj
                data['LQs'] = [utils.decode_jpeg(lr) for lr in data['LQs']]
                data['GT'] = [utils.decode_jpeg(hr) for hr in data['GT']]
                data['LQs'] = np.stack(data['LQs'], axis=0)
                data['GT'] = np.stack(data['GT'], axis=0)
                data['LQs'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['LQs'], (0, 3, 1, 2)))).float()
                data['GT'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['GT'], (0, 3, 1, 2)))).float()
                data['key'] = time.time()
                #准备数据
                sample_cnt = sample_cnt + 1
                samples_lq.append(data['LQs'])
                samples_gt.append(data['GT'])
                if sample_cnt == self.batch_size:#样本达到一定数量后开始训练
                    sample_cnt = 0
                    data['GT'] = torch.stack(samples_gt, dim=0) / 255.  # to0-1
                    data['LQs'] = torch.stack(samples_lq, dim=0) / 255.
                    self.sr_model.learning(data)
                    samples_lq.clear()
                    samples_gt.clear()
        self.is_terminated=True
        self.logger.info('client ' + str(self.client_sock.getpeername()) + ' disconnected')