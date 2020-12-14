import torch
from src.solver import BaseSolver

from src.lm import RNNLM
from src.optim import Optimizer
from src.data import load_textset
from src.util import human_format


class Solver(BaseSolver):
    ''' Solver for training language models'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        # Logger settings
        self.best_loss = 10

    def fetch_data(self, data):
        ''' Move data to device, insert <sos> and compute text seq. length'''
        txt = torch.cat((torch.zeros((data.shape[0],1),dtype=torch.long),data), dim=1).to(self.device)
        txt_len = torch.sum(data!=0,dim=-1)
        return txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.vocab_size, self.tokenizer, msg = \
                         load_textset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''

        # Model
        self.model = RNNLM( self.vocab_size, **self.config['model']).to(self.device)
        self.verbose(self.model.create_msg())
        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer
        self.optimizer = Optimizer(self.model.parameters(),**self.config['hparas'])
        # Enable AMP if needed
        self.enable_apex()
        # load pre-trained model
        if self.paras.load:
            self.load_ckpt()
            ckpt = torch.load(self.paras.load, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_opt_state_dict(ckpt['optimizer'])
            self.step = ckpt['global_step']
            self.verbose('Load ckpt from {}, restarting at step {}'.format(self.paras.load,self.step))

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        self.timer.set()
        
        while self.step< self.max_step:
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                self.optimizer.pre_step(self.step)

                # Fetch data
                txt, txt_len = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                pred, _ = self.model(txt[:,:-1], txt_len)

                # Compute all objectives
                lm_loss = self.seq_loss(pred.view(-1,self.vocab_size),txt[:,1:].reshape(-1))
                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(lm_loss)
                self.step +=1

                # Logger
                if self.step%self.PROGRESS_STEP==0:
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(lm_loss.cpu().item(),grad_norm,self.timer.show()))
                    self.write_log('entropy',{'tr':lm_loss})
                    self.write_log('perplexity',{'tr':torch.exp(lm_loss).cpu().item()})
                
                # Validation
                if (self.step==1) or (self.step%self.valid_step == 0):
                    self.validate()

                # End of step
                self.timer.set()
                if self.step > self.max_step:break
        self.log.close()
    
    def validate(self):
        # Eval mode
        self.model.eval()
        dev_loss = []

        for i,data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(self.dv_set)))
            # Fetch data
            txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                pred, _ = self.model(txt[:,:-1], txt_len)
            lm_loss = self.seq_loss(pred.view(-1,self.vocab_size),txt[:,1:].reshape(-1))
            dev_loss.append(lm_loss)
        
        # Ckpt if performance improves
        dev_loss = sum(dev_loss)/len(dev_loss)
        dev_ppx = torch.exp(dev_loss).cpu().item()
        if dev_loss < self.best_loss :
            self.best_loss = dev_loss
            self.save_checkpoint('best_ppx.pth','perplexity',dev_ppx)
        self.write_log('entropy',{'dv':dev_loss})
        self.write_log('perplexity',{'dv':dev_ppx})

        # Show some example of last batch on tensorboard
        for i in range(min(len(txt),self.DEV_N_EXAMPLE)):
            if self.step ==1:
                self.write_log('true_text{}'.format(i),self.tokenizer.decode(txt[i].tolist()))
            self.write_log('pred_text{}'.format(i),self.tokenizer.decode(pred[i].argmax(dim=-1).tolist()))

        # Resume training
        self.model.train()
