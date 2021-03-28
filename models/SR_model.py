import copy
import logging
import threading
import time
import torch
import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.loss as loss


class SRModel():
    def __init__(self, model_path, learning_rate=0.01, update_interval=5):
        if torch.cuda.is_available():
            self.pred_device = torch.device('cuda:0')#used for one GPU
            self.multi_gpu_flag = False
        else:
            self.pred_device = torch.device('cpu')
            self.multi_gpu_flag = False
        # 根据GPU情况初始化模型
        self.logger=logging.getLogger('base')#获取logger
        self.update_interval = update_interval
        self.model_update_time = None  # 记录上次更新时间，过了一定时长后更新
        #定义模型更新参数
        model = Sakuya_arch.LunaTokis(64, 3, 8, 5, 40)
        model.load_state_dict(torch.load(str(model_path)), strict=True)
        self.pred_model = model.to(self.pred_device)
        #self.pred_model=torch.nn.DataParallel(model,device_ids=[0,1])
        self.pred_model.eval()
        #定义模型
        self.critic = loss.CharbonnierLoss().to(self.pred_device)
        opt_params = []
        for k, v in self.pred_model.named_parameters():
            if v.requires_grad:
                opt_params.append(v)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(opt_params, self.learning_rate)
        #定义optimizer和损失函数

    def learning(self, data):
        self.pred_model.train()
        self.optimizer.zero_grad()
        output = self.pred_model(data['LQs'].to(self.pred_device))
        loss = self.critic(output, data['GT'].to(self.pred_device))
        self.logger.info('time: {} ;loss is: {}'.format(data['key'], loss.cpu().item()))
        loss.backward()
        self.optimizer.step()
        if self.model_update_time is None:
            self.model_update_time = time.time()
        elif time.time() - self.model_update_time > self.update_interval:
            self.update_model()
            self.model_update_time = time.time()
        return loss

    def update_model(self):
        pass

    def forward(self, data):
        self.pred_model.eval()
        output = self.pred_model(data.to(self.pred_device))
        return output

