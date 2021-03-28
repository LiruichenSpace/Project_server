import logging
import threading
import time
import matplotlib.pyplot as plt

import cv2
import torch

import network
import utils
import numpy as np
import os.path as osp
from models.SR_model import SRModel

def process_and_save(model,input,save_path,start_index,obj,logger,start_time):
    model.pred_model.eval()
    with torch.no_grad():
        output = model.forward(input)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    output = output.data.float().cpu()

    # TODO handle output data here
    # self.logger.info('output shape:'+str(output.shape))
    output = output[:, :, :, 0:obj['shape'][1], 0:obj['shape'][0]]  # cut the frames to original size

    for i in range(2):
        tmp = output[0, i + 1, :, :, :].squeeze(0).squeeze(0).numpy()
        tmp = tmp.transpose((1, 2, 0))[0:obj['shape'][1], 0:obj['shape'][0], ::-1]
        #cv2.imwrite(osp.join(save_path,'{}.png'.format(start_index)), tmp * 255.)
        start_index += 1
    logger.info(
        "solved frames:{} , speed:{}/s".format(start_index, start_index / (time.time() - start_time)))

def process_and_learn(sr_model, data, samples_lq, samples_gt,batch_size):
    data['LQs'] = [utils.decode_jpeg(lr) for lr in data['LQs']]
    data['GT'] = [utils.decode_jpeg(hr) for hr in data['GT']]
    data['LQs'] = np.stack(data['LQs'], axis=0)
    data['GT'] = np.stack(data['GT'], axis=0)
    data['LQs'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['LQs'], (0, 3, 1, 2)))).float()
    data['GT'] = torch.from_numpy(np.ascontiguousarray(np.transpose(data['GT'], (0, 3, 1, 2)))).float()
    data['key'] = time.time()
    # 准备数据
    samples_lq.append(data['LQs'])
    samples_gt.append(data['GT'])
    if len(samples_gt) == batch_size:  # 样本达到一定数量后开始训练
        sample_cnt = 0
        data['GT'] = torch.stack(samples_gt, dim=0) / 255.  # to0-1
        data['LQs'] = torch.stack(samples_lq, dim=0) / 255.
        sr_model.learning(data)
        samples_lq.clear()
        samples_gt.clear()

class ClientHandler(threading.Thread):
    #TODO 添加线程对外开放的变量，用来进行调度处理，如模型表现等等
    def __init__(self, csock,sr_model,batch_size=3,outframe_cnt=2):
        super().__init__()
        self.client_sock=csock
        self.sr_model=sr_model
        self.terminate=False    #主动停止，用于控制
        self.batch_size=batch_size
        self.logger=logging.getLogger('base')
        self.count=1
        self.start_time=time.time()
        self.is_terminated=False#状态变量，用于判断是否结束
        self.out_frame_cnt=outframe_cnt
    def run(self) -> None:
        plt.ion()
        samples_lq=[]
        samples_gt=[]
        sample_cnt=0
        frame_list=[]
        prev=None
        while not self.terminate:
            is_sample,obj=network.handel_stream(self.client_sock)
            if not is_sample and obj is None:
                self.terminate=True
                break
            if not is_sample:
                #self.logger.info(obj['shape'])
                # plt.imshow(data)
                # plt.draw()
                # plt.show()
                #resized = cv2.resize(data, (data.shape[1] * 4, data.shape[0] * 4), cv2.INTER_NEAREST) / 255.0
                curr = utils.decode_jpeg(obj['img'])
                curr = curr.transpose((2, 0, 1)) /255. # because the original data of jpeg decode is 0-255,should
                # be scaled to 0-1
                if len(frame_list)<self.out_frame_cnt//2:
                    frame_list.append(curr)
                    continue
                else:
                    frame_list.append(curr)
                    data = np.stack(frame_list, axis=0)
                    frame_list=[curr]
                data = torch.from_numpy(np.ascontiguousarray(data)).float()
                data = data.unsqueeze(0)
                #self.logger.info(data.shape)

                # use multythread here
                thread=threading.Thread(target=process_and_save(self.sr_model
                                                                ,data
                                                                ,'/home/wsn/LRC_div/results'
                                                                ,self.count
                                                                ,obj
                                                                ,self.logger
                                                                ,self.start_time))
                thread.setDaemon(True)
                thread.start()
                self.count+=2


                # self.sr_model.pred_model.eval()
                # with torch.no_grad():
                #     tmp=time.time()
                #     output = self.sr_model.forward(data)
                #     self.logger.info("forward use time:{} s\n".format(time.time()-tmp))
                # if isinstance(output, list) or isinstance(output, tuple):
                #     output = output[0]
                # output = output.data.float().cpu()
                #
                # #TODO handle output data here
                # #self.logger.info('output shape:'+str(output.shape))
                # output=output[:,:,:,0:obj['shape'][1],0:obj['shape'][0]]#cut the frames to original size
                #
                # for i in range(6):
                #     tmp=output[0,i+1,:,:,:].squeeze(0).squeeze(0).numpy()
                #     tmp=tmp.transpose((1,2,0))[0:obj['shape'][1],0:obj['shape'][0],::-1]
                #     #cv2.imwrite('/home/wsn/LRC_div/results/{}.png'.format(self.count),tmp*255.)
                #     self.count=self.count+1
                # self.logger.info(
                #     "solved frames:{} , speed:{}/s".format(self.count, self.count / (time.time() - self.start_time)))



                #     plt.imshow(tmp)
                #     plt.draw()
                #     plt.show()

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

                thread=threading.Thread(target=process_and_learn(self.sr_model
                                                                ,obj
                                                                ,samples_lq
                                                                ,samples_gt
                                                                ,self.batch_size))
                thread.setDaemon(True)
                thread.start()
                continue
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