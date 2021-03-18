import logging
import threading
import time
import torch
import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.loss as loss
from models.SR_model import SRModel


class SRModelShared(SRModel):
    def __init__(self, model_path, learning_rate=0.01, update_interal=4):
        super().__init__(model_path, learning_rate, update_interal)
        if torch.cuda.is_available():
            self.train_device = torch.device('cuda:0')
            self.learning_lock = threading.Lock()  # 用来避免同时进行学习，如果模型估计不能并发进行的话，可能调用模型也需要加锁
            self.pred_lock = self.learning_lock  # 如果有多个GPU，此处可以修改以提升效率
        else:
            self.train_device = torch.device('cpu')
            self.learning_lock = threading.Lock()
            self.pred_lock = self.learning_lock
        model = Sakuya_arch.LunaTokis(64, 3, 8, 5, 40)
        model.load_state_dict(torch.load('./test/saved_model.pth'), strict=True)
        self.train_model = model.to(self.train_device)
        self.train_model.train()
        #定义模型
        self.critic = loss.CharbonnierLoss().to(self.train_device)
        opt_params = []
        for k, v in self.pred_model.named_parameters():
            if v.requires_grad:
                opt_params.append(v)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(opt_params, self.learning_rate)
        #定义optimizer和损失函数

    def learning(self, data):
        with self.learning_lock:  # 互斥学习
            self.optimizer.zero_grad()
            output = self.train_model(data['LQs'].to(self.train_device))
            loss = self.critic(output, data['GT'].to(self.train_device))
            self.logger.info('time: {} ;loss is: {}'.format(data['key'], loss.cpu().item()))
            loss.backward()
            self.optimizer.step()
            if self.model_update_time is None:
                self.model_update_time = time.time()
            elif time.time() - self.model_update_time > self.update_interval:
                self.update_model()
                self.model_update_time = time.time()

    def update_model(self):
        if self.multi_gpu_flag:
            #with  self.pred_lock:
            self.pred_model = self.train_model.to(self.pred_device)
            self.pred_model.eval()
        else:
            #with  self.pred_lock:
            self.pred_model = self.train_model.cpu().to(self.pred_device)
            self.pred_model.eval()
            # self.pred_model=copy.deepcopy(self.train_model)    #目的是拷贝模型，可行性未知

    def forward(self, data):
        #此处是否可以去掉这个锁？
        with self.pred_lock:
            output = self.pred_model(data.to(self.pred_device))
        return output

