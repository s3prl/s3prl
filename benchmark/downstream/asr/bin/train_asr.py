import torch
import torch.nn as nn
import yaml

from src.solver import BaseSolver

from src.asr import ASR
from src.optim import Optimizer
from src.data import load_dataset
from src.util import human_format, cal_er, feat_to_fig, LabelSmoothingLoss
from src.audio import Delta, Postprocess, Augment

EMPTY_CACHE_STEP = 100

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)

        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']
        self.val_mode = self.config['hparas']['val_mode'].lower()
        self.WER = 'per' if self.val_mode == 'per' else 'wer'

    def fetch_data(self, data, train=False):
        ''' Move data to device and compute text seq. length'''
        # feat: B x T x D
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        
        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      self.curriculum>0,
                                      **self.config['data'])
        self.verbose(msg)

        # Dev set sames
        self.dv_names = []
        if type(self.dv_set) is list:
            for ds in self.config['data']['corpus']['dev_split']:
                self.dv_names.append(ds[0])
        else:
            self.dv_names = self.config['data']['corpus']['dev_split'][0]
        
        # Logger settings
        if type(self.dv_names) is str:
            self.best_wer = {'att':{self.dv_names:3.0},
                             'ctc':{self.dv_names:3.0}}
        else:
            self.best_wer = {'att': {},'ctc': {}}
            for name in self.dv_names:
                self.best_wer['att'][name] = 3.0
                self.best_wer['ctc'][name] = 3.0

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        #print(self.feat_dim) #160
        batch_size = self.config['data']['corpus']['batch_size']//2
        self.model = ASR(self.feat_dim, self.vocab_size, batch_size, **self.config['model']).to(self.device)



        self.verbose(self.model.create_msg())
        model_paras = [{'params':self.model.parameters()}]

        # Losses
        
        '''label smoothing'''
        if self.config['hparas']['label_smoothing']:
            self.seq_loss = LabelSmoothingLoss(31, 0.1)   
            print('[INFO]  using label smoothing. ') 
        else:    
            self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False) # Note: zero_infinity=False is unstable?

        # Plug-ins
        self.emb_fuse = False
        self.emb_reg = ('emb' in self.config) and (self.config['emb']['enable'])
        if self.emb_reg:
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(self.tokenizer, self.model.dec_dim, **self.config['emb']).to(self.device)
            model_paras.append({'params':self.emb_decoder.parameters()})
            self.emb_fuse = self.emb_decoder.apply_fuse
            if self.emb_fuse:
                self.seq_loss = torch.nn.NLLLoss(ignore_index=0)
            self.verbose(self.emb_decoder.create_msg())

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.lr_scheduler = self.optimizer.lr_scheduler
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()
        
        # Transfer Learning
        if self.transfer_learning:
            self.verbose('Apply transfer learning: ')
            self.verbose('      Train encoder layers: {}'.format(self.train_enc))
            self.verbose('      Train decoder:        {}'.format(self.train_dec))
            self.verbose('      Save name:            {}'.format(self.save_name))
        
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        if self.transfer_learning:
            self.model.encoder.fix_layers(self.fix_enc)
            if self.fix_dec and self.model.enable_att:
                self.model.decoder.fix_layers()
            if self.fix_dec and self.model.enable_ctc:
                self.model.fix_ctc_layer()
        
        self.n_epochs = 0
        self.timer.set()
        '''early stopping for ctc '''
        self.early_stoping = self.config['hparas']['early_stopping']
        stop_epoch = 10
        batch_size = self.config['data']['corpus']['batch_size']
        stop_step = len(self.tr_set)*stop_epoch//batch_size
        


        while self.step< self.max_step:
            ctc_loss, att_loss, emb_loss = None, None, None
            # Renew dataloader to enable random sampling 
            
            if self.curriculum>0 and n_epochs==self.curriculum:
                self.verbose('Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      False, **self.config['data'])
            
            
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)
                total_loss = 0
                
                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(data, train=True)
            
                self.timer.cnt('rd')
                # Forward model
                # Note: txt should NOT start w/ <sos>
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model( feat, feat_len, max(txt_len), tf_rate=tf_rate,
                                    teacher=txt, get_dec_state=self.emb_reg)
                # Clear not used objects
                del att_align

                # Plugins
                if self.emb_reg:
                    emb_loss, fuse_output = self.emb_decoder( dec_state, att_output, label=txt) 
                    total_loss += self.emb_decoder.weight*emb_loss
                else:
                    del dec_state

                ''' early stopping ctc'''
                if self.early_stoping:
                    if self.step > stop_step:
                        ctc_output = None
                        self.model.ctc_weight = 0
                #print(ctc_output.shape)
                # Compute all objectives
                if ctc_output is not None:
                    if self.paras.cudnn_ctc:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), 
                                                 txt.to_sparse().values().to(device='cpu',dtype=torch.int32),
                                                 [ctc_output.shape[1]]*len(ctc_output),
                                                 #[int(encode_len.max()) for _ in encode_len],
                                                 txt_len.cpu().tolist())
                    else:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), txt, encode_len, txt_len)
                    total_loss += ctc_loss*self.model.ctc_weight
                    del encode_len

                if att_output is not None:
                    #print(att_output.shape)
                    b,t,_ = att_output.shape
                    att_output = fuse_output if self.emb_fuse else att_output
                    att_loss = self.seq_loss(att_output.view(b*t,-1),txt.view(-1))
                    # Sum each uttr and devide by length then mean over batch
                    # att_loss = torch.mean(torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(txt!=0,dim=-1).float())
                    total_loss += att_loss*(1-self.model.ctc_weight)

                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)             

                self.step+=1
                
                # Logger
                if (self.step==1) or (self.step%self.PROGRESS_STEP==0):
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(total_loss.cpu().item(),grad_norm,self.timer.show()))
                    self.write_log('emb_loss',{'tr':emb_loss})
                    if att_output is not None:
                        self.write_log('loss',{'tr_att':att_loss})
                        self.write_log(self.WER,{'tr_att':cal_er(self.tokenizer,att_output,txt)})
                        self.write_log(   'cer',{'tr_att':cal_er(self.tokenizer,att_output,txt,mode='cer')})
                    if ctc_output is not None:
                        self.write_log('loss',{'tr_ctc':ctc_loss})
                        self.write_log(self.WER,{'tr_ctc':cal_er(self.tokenizer,ctc_output,txt,ctc=True)})
                        self.write_log(   'cer',{'tr_ctc':cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True)})
                        self.write_log('ctc_text_train',self.tokenizer.decode(ctc_output[0].argmax(dim=-1).tolist(),
                                                                                                ignore_repeat=True))
                    # if self.step==1 or self.step % (self.PROGRESS_STEP * 5) == 0:
                    #     self.write_log('spec_train',feat_to_fig(feat[0].transpose(0,1).cpu().detach(), spec=True))
                    #del total_loss
                    
                    if self.emb_fuse: 
                        if self.emb_decoder.fuse_learnable:
                            self.write_log('fuse_lambda',{'emb':self.emb_decoder.get_weight()})
                        self.write_log('fuse_temp',{'temp':self.emb_decoder.get_temp()})

                # Validation
                if (self.step==1) or (self.step%self.valid_step == 0):
                    if type(self.dv_set) is list:
                        for dv_id in range(len(self.dv_set)):
                            self.validate(self.dv_set[dv_id], self.dv_names[dv_id])
                    else:
                        self.validate(self.dv_set, self.dv_names)
                if self.step % (len(self.tr_set)// batch_size)==0: # one epoch
                    print('Have finished epoch: ', self.n_epochs)
                    self.n_epochs +=1
                    
                if self.lr_scheduler == None:
                    lr = self.optimizer.opt.param_groups[0]['lr']
                    
                    if self.step == 1:
                        print('[INFO]    using lr schedular defined by Daniel, init lr = ', lr)

                    if self.step >99999 and self.step%2000==0:
                        lr = lr*0.85
                        for param_group in self.optimizer.opt.param_groups:
                            param_group['lr'] = lr
                        print('[INFO]     at step:', self.step )
                        print('[INFO]   lr reduce to', lr)


                    #self.lr_scheduler.step(total_loss)
                # End of step
                # if self.step % EMPTY_CACHE_STEP == 0:
                    # Empty cuda cache after every fixed amount of steps
                torch.cuda.empty_cache() # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                self.timer.set()
                if self.step > self.max_step: break
            
            
            
            #update lr_scheduler
            
            
        self.log.close()
        print('[INFO] Finished training after', human_format(self.max_step), 'steps.')
        
    def validate(self, _dv_set, _name):
        # Eval mode
        self.model.eval()
        if self.emb_decoder is not None: self.emb_decoder.eval()
        dev_wer = {'att':[],'ctc':[]}
        dev_cer = {'att':[],'ctc':[]}
        dev_er  = {'att':[],'ctc':[]}

        for i,data in enumerate(_dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(_dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model( feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO), 
                                    emb_decoder=self.emb_decoder)

            if att_output is not None:
                dev_wer['att'].append(cal_er(self.tokenizer,att_output,txt,mode='wer'))
                dev_cer['att'].append(cal_er(self.tokenizer,att_output,txt,mode='cer'))
                dev_er['att'].append(cal_er(self.tokenizer,att_output,txt,mode=self.val_mode))
            if ctc_output is not None:
                dev_wer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode='wer',ctc=True))
                dev_cer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True))
                dev_er['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode=self.val_mode,ctc=True))
            
            # Show some example on tensorboard
            if i == len(_dv_set)//2:
                for i in range(min(len(txt),self.DEV_N_EXAMPLE)):
                    if self.step==1:
                        self.write_log('true_text_{}_{}'.format(_name, i),self.tokenizer.decode(txt[i].tolist()))
                    if att_output is not None:
                        self.write_log('att_align_{}_{}'.format(_name, i),feat_to_fig(att_align[i,0,:,:].cpu().detach()))
                        self.write_log('att_text_{}_{}'.format(_name, i),self.tokenizer.decode(att_output[i].argmax(dim=-1).tolist()))
                    if ctc_output is not None:
                        self.write_log('ctc_text_{}_{}'.format(_name, i),self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
                                                                                                       ignore_repeat=True))
        
        # Ckpt if performance improves
        tasks = []
        if len(dev_er['att']) > 0:
            tasks.append('att')
        if len(dev_er['ctc']) > 0:
            tasks.append('ctc')

        for task in tasks:
            dev_er[task] = sum(dev_er[task])/len(dev_er[task])
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            dev_cer[task] = sum(dev_cer[task])/len(dev_cer[task])
            if dev_er[task] < self.best_wer[task][_name]:
                self.best_wer[task][_name] = dev_er[task]
                self.save_checkpoint('best_{}_{}.pth'.format(task, _name + (self.save_name if self.transfer_learning else '')), 
                                    self.val_mode,dev_er[task],_name)
            if self.step >= self.max_step:
                self.save_checkpoint('last_{}_{}.pth'.format(task, _name + (self.save_name if self.transfer_learning else '')), 
                                    self.val_mode,dev_er[task],_name)
            self.write_log(self.WER,{'dv_'+task+'_'+_name.lower():dev_wer[task]})
            self.write_log(   'cer',{'dv_'+task+'_'+_name.lower():dev_cer[task]})
            # if self.transfer_learning:
            #     print('[{}] WER {:.4f} / CER {:.4f} on {}'.format(human_format(self.step), dev_wer[task], dev_cer[task], _name))

        # Resume training
        self.model.train()
        if self.transfer_learning:
            self.model.encoder.fix_layers(self.fix_enc)
            if self.fix_dec and self.model.enable_att:
                self.model.decoder.fix_layers()
            if self.fix_dec and self.model.enable_ctc:
                self.model.fix_ctc_layer()
        
        if self.emb_decoder is not None: self.emb_decoder.train()
