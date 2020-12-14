import torch
import numpy as np
import torch.optim.lr_scheduler as LR
class Optimizer():
    def __init__(self, parameters, optimizer, lr, eps, lr_scheduler, 
                    tf_start=1, tf_end=1, tf_step=1, tf_step_start=0, 
                    weight_decay=0, amsgrad=False, **kwargs):
        
        # Setup teacher forcing scheduler
        self.tf_type = tf_end!=1
        self.tf_rate = lambda step: max(tf_end, 
            tf_start-(tf_start-tf_end)*(step-tf_step_start)/tf_step if step >= tf_step_start else 1)

        # Setup torch optimizer
        self.opt_type = optimizer
        self.init_lr = lr
        self.sch_type = lr_scheduler
        opt = getattr(torch.optim,optimizer)
        if lr_scheduler == 'warmup':
            warmup_step = 4000.0
            init_lr = lr
            self.lr_scheduler = lambda step: init_lr * warmup_step **0.5 * np.minimum((step+1)*warmup_step**-1.5,(step+1)**-0.5 )
            self.opt = opt(parameters,lr=1.0) 
        else:
            self.lr_scheduler = None
            if optimizer.lower()[:4] == 'adam':
                self.opt = opt(parameters,lr=lr,eps=eps,weight_decay=weight_decay,amsgrad=amsgrad) # ToDo: 1e-8 better?
            else:
                self.opt = opt(parameters,lr=lr,eps=eps,weight_decay=weight_decay) # ToDo: 1e-8 better?

    def get_opt_state_dict(self):
        return self.opt.state_dict()

    def load_opt_state_dict(self,state_dict):
        self.opt.load_state_dict(state_dict)

    def pre_step(self, step):
        '''
        if self.lr_scheduler is not None:
            cur_lr = self.lr_scheduler(step)
            for param_group in self.opt.param_groups:
                param_group['lr'] = cur_lr
        '''
        self.opt.zero_grad()
        return self.tf_rate(step)
    
    def get_lr(self, step):
        if self.lr_scheduler is not None:
            return self.lr_scheduler(step)
        else:
            return self.init_lr

    def step(self):
        self.opt.step()

    def create_msg(self):
        return ['Optim.spec.| Algo. = {}\t| Lr = {}\t (schedule = {})| Scheduled sampling = {}'\
                   .format(self.opt_type, self.init_lr, self.sch_type, self.tf_type)]
