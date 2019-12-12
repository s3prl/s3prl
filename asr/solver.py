# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ asr/solver.py ]
#   Synopsis     [ solver for asr]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import copy
import math
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from joblib import Parallel, delayed
from tensorboardX import SummaryWriter
from asr.rnnlm import RNN_LM
from asr.model import Seq2Seq
from asr.clm import CLM_wrapper
from dataloader import get_Dataloader
from utility.asr import Mapper, cal_acc, cal_cer, draw_att


VAL_STEP = 30        # Additional Inference Timesteps to run during validation (to calculate CER)
TRAIN_WER_STEP = 250 # steps for debugging info.
GRAD_CLIP = 5
CLM_MIN_SEQ_LEN = 5


class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self, config, paras):
        # General Settings
        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        if not os.path.exists(paras.ckpdir):os.makedirs(paras.ckpdir)
        self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
        if not os.path.exists(self.ckpdir):os.makedirs(self.ckpdir)
        
        # Load Mapper for idx2token
        self.mapper = Mapper(config['solver']['data_path'])

        if torch.cuda.is_available(): self.verbose('CUDA is available!')

    def verbose(self, msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[SOLVER]', msg)
   
    def progress(self, msg):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose:
            print(msg + '                              ', end='\r')


class Trainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras):
        super(Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir, self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_val_ed = 2.0

        # Training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        self.tf_start = config['solver']['tf_start']
        self.tf_end = config['solver']['tf_end']
        self.apex = config['solver']['apex']

        # CLM option
        self.apply_clm = config['clm']['enable']

    def load_data(self):
        ''' Load date for training/validation'''
        self.verbose('Loading data from ' + self.config['solver']['data_path'])
        setattr(self, 'train_set', get_Dataloader('train', load='asr', use_gpu=self.paras.gpu, **self.config['solver']))
        setattr(self, 'dev_set', get_Dataloader('dev',load='asr', use_gpu=self.paras.gpu, **self.config['solver']))
        
        # Get 1 example for auto constructing model
        for self.sample_x, _ in getattr(self,'train_set'): break
        if len(self.sample_x.shape) == 4: self.sample_x = self.sample_x[0]

    def set_model(self):
        ''' Setup ASR (and CLM if enabled)'''
        self.verbose('Init ASR model. Note: validation is done through greedy decoding w/ attention decoder.')
        
        # Build attention end-to-end ASR
        self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model']).to(self.device)
        if 'VGG' in self.config['asr_model']['encoder']['enc_type']:
            self.verbose('VCC Extractor in Encoder is enabled, time subsample rate = 4.')
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')
        
        # Involve CTC
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']
        
        # TODO: load pre-trained model
        if self.paras.load:
            raise NotImplementedError
            
        # Setup optimizer
        if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
            import apex
            self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])
        else:
            self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)

        # Apply CLM
        if self.apply_clm:
            self.clm = CLM_wrapper(self.mapper.get_dim(), self.config['clm']).to(self.device)
            clm_data_config = self.config['solver']
            clm_data_config['train_set'] = self.config['clm']['source']
            clm_data_config['use_gpu'] = self.paras.gpu
            self.clm.load_text(clm_data_config)
            self.verbose('CLM is enabled with text-only source: '+str(clm_data_config['train_set']))
            self.verbose('Extra text set total '+str(len(self.clm.train_set))+' batches.')

    def exec(self):
        ''' Training End-to-end ASR system'''
        self.verbose('Training set total '+str(len(self.train_set))+' batches.')

        while self.step< self.max_step:
            for x,y in self.train_set:
                self.progress('Training step - '+str(self.step))
                
                # Perform teacher forcing rate decaying
                tf_rate = self.tf_start - self.step*(self.tf_start-self.tf_end)/self.max_step
                
                # Hack bucket, record state length for each uttr, get longest label seq for decode step
                assert len(x.shape)==4,'Bucketing should cause acoustic feature to have shape 1xBxTxD'
                assert len(y.shape)==3,'Bucketing should cause label have to shape 1xBxT'
                x = x.squeeze(0).to(device = self.device,dtype=torch.float32)
                y = y.squeeze(0).to(device = self.device,dtype=torch.long)
                state_len = np.sum(np.sum(x.cpu().data.numpy(),axis=-1)!=0,axis=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # ASR forwarding 
                self.asr_opt.zero_grad()
                ctc_pred, state_len, att_pred, _ =  self.asr_model(x, ans_len,tf_rate=tf_rate,teacher=y,state_len=state_len)

                # Calculate loss function
                loss_log = {}
                label = y[:,1:ans_len+1].contiguous()
                ctc_loss = 0
                att_loss = 0
                
                # CE loss on attention decoder
                if self.ctc_weight<1:
                    b,t,c = att_pred.shape
                    att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))
                    att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                               .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                    att_loss = torch.mean(att_loss) # Mean by batch
                    loss_log['train_att'] = att_loss

                # CTC loss on CTC decoder
                if self.ctc_weight>0:
                    target_len = torch.sum(y!=0,dim=-1)
                    ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
                    loss_log['train_ctc'] = ctc_loss
                
                asr_loss = (1-self.ctc_weight)*att_loss+self.ctc_weight*ctc_loss
                loss_log['train_full'] = asr_loss
                
                # Adversarial loss from CLM
                if self.apply_clm and att_pred.shape[1]>=CLM_MIN_SEQ_LEN:
                    if (self.step%self.clm.update_freq)==0:
                        # update CLM once in a while
                        clm_log,gp = self.clm.train(att_pred.detach(),CLM_MIN_SEQ_LEN)
                        self.write_log('clm_score',clm_log)
                        self.write_log('clm_gp',gp)
                    adv_feedback = self.clm.compute_loss(F.softmax(att_pred))
                    asr_loss -= adv_feedback

                # Backprop
                asr_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step '+str(self.step))
                else:
                    self.asr_opt.step()
                
                # Logger
                self.write_log('loss',loss_log)
                if self.ctc_weight<1:
                    self.write_log('acc',{'train':cal_acc(att_pred,label)})
                if self.step % TRAIN_WER_STEP ==0:
                    self.write_log('error rate',
                                   {'train':cal_cer(att_pred,label,mapper=self.mapper)})

                # Validation
                if self.step % self.valid_step == 0 and self.step != 0:
                    self.asr_opt.zero_grad()
                    self.valid()

                self.step+=1
                if self.step > self.max_step:break
    

    def write_log(self,val_name,val_dict):
        '''Write log to TensorBoard'''
        if 'att' in val_name:
            self.log.add_image(val_name,val_dict,self.step)
        elif 'txt' in val_name or 'hyp' in val_name:
            self.log.add_text(val_name, val_dict, self.step)
        else:
            self.log.add_scalars(val_name,val_dict,self.step)


    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding with Attention decoder only)'''
        self.asr_model.eval()
        
        # Init stats
        val_loss, val_ctc, val_att, val_acc, val_cer = 0.0, 0.0, 0.0, 0.0, 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        
        # Perform validation
        for cur_b,(x,y) in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(self.step),'(',str(cur_b),'/',str(len(self.dev_set)),')']))

            # Prepare data
            if len(x.shape)==4: x = x.squeeze(0)
            if len(y.shape)==3: y = y.squeeze(0)
            x = x.to(device = self.device,dtype=torch.float32)
            y = y.to(device = self.device,dtype=torch.long)
            state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
            
            # Forward
            ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

            # Compute attention loss & get decoding results
            label = y[:,1:ans_len+1].contiguous()
            if self.ctc_weight<1:
                seq_loss = self.seq_loss(att_pred[:,:ans_len,:].contiguous().view(-1,att_pred.shape[-1]),label.view(-1))
                seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)/torch.sum(y!=0,dim=-1)\
                           .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                seq_loss = torch.mean(seq_loss) # Mean by batch
                val_att += seq_loss.detach()*int(x.shape[0])
                t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_acc += cal_acc(att_pred,label)*int(x.shape[0])
                val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
            
            # Compute CTC loss
            if self.ctc_weight>0:
                target_len = torch.sum(y!=0,dim=-1)
                ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, 
                                         torch.LongTensor(state_len), target_len)
                val_ctc += ctc_loss.detach()*int(x.shape[0])

            val_len += int(x.shape[0])
        
        # Logger
        val_loss = (1-self.ctc_weight)*val_att + self.ctc_weight*val_ctc
        loss_log = {}
        for k,v in zip(['dev_full','dev_ctc','dev_att'],[val_loss, val_ctc, val_att]):
            if v > 0.0: loss_log[k] = v/val_len
        self.write_log('loss',loss_log)
 
        if self.ctc_weight<1:
            # Plot attention map to log
            val_hyp,val_txt = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
            val_attmap = draw_att(att_maps,att_pred)

            # Record loss
            self.write_log('error rate',{'dev':val_cer/val_len})
            self.write_log('acc',{'dev':val_acc/val_len})
            for idx,attmap in enumerate(val_attmap):
                self.write_log('att_'+str(idx),attmap)
                self.write_log('hyp_'+str(idx),val_hyp[idx])
                self.write_log('txt_'+str(idx),val_txt[idx])

            # Save model by val er.
            if val_cer/val_len  < self.best_val_ed:
                self.best_val_ed = val_cer/val_len
                self.verbose('Best val er       : {:.4f}       @ step {}'.format(self.best_val_ed,self.step))
                torch.save(self.asr_model, os.path.join(self.ckpdir,'asr'))
                if self.apply_clm:
                    torch.save(self.clm.clm,  os.path.join(self.ckpdir,'clm'))
                # Save hyps.
                with open(os.path.join(self.ckpdir,'best_hyp.txt'),'w') as f:
                    for t1,t2 in zip(all_pred,all_true):
                        f.write(t1+','+t2+'\n')

        self.asr_model.train()


class Tester(Solver):
    ''' Handler for complete inference progress'''
    def __init__(self,config,paras):
        super(Tester, self).__init__(config,paras)
        self.verbose('During beam decoding, batch size is set to 1, please speed up with --njobs.')
        self.njobs = self.paras.njobs
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']
        
        self.decode_file = "_".join(['decode','beam',str(self.config['solver']['decode_beam_size']),
                                     'len',str(self.config['solver']['max_decode_step_ratio'])])

    def load_data(self):
        self.verbose('Loading testing data '+str(self.config['solver']['test_set'])\
                     +' from '+self.config['solver']['data_path'])
        setattr(self,'test_set', get_Dataloader('test', load='asr', use_gpu=self.paras.gpu, **self.config['solver']))
        setattr(self,'dev_set', get_Dataloader('dev', load='asr', use_gpu=self.paras.gpu, **self.config['solver']))

    def set_model(self):
        ''' Load saved ASR'''
        self.verbose('Load ASR model from '+os.path.join(self.ckpdir))
        self.asr_model = torch.load(os.path.join(self.ckpdir,'asr'))
        
        # Enable joint CTC decoding
        self.asr_model.joint_ctc = self.config['solver']['decode_ctc_weight'] >0
        if self.config['solver']['decode_ctc_weight'] >0:
            assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
            self.verbose('Joint CTC decoding is enabled with weight = '+str(self.config['solver']['decode_ctc_weight']))
            self.decode_file += '_ctc{:}'.format(self.config['solver']['decode_ctc_weight'])
            self.asr_model.ctc_weight = self.config['solver']['decode_ctc_weight']
            
        # Enable joint RNNLM decoding
        self.decode_lm = self.config['solver']['decode_lm_weight'] >0
        setattr(self.asr_model,'decode_lm_weight',self.config['solver']['decode_lm_weight'])
        if self.decode_lm:
            assert os.path.exists(self.config['solver']['decode_lm_path']), 'Please specify RNNLM path.'
            self.asr_model.load_lm(**self.config['solver'])
            self.verbose('Joint RNNLM decoding is enabled with weight = '+str(self.config['solver']['decode_lm_weight']))
            self.verbose('Loading RNNLM from '+self.config['solver']['decode_lm_path'])
            self.decode_file += '_lm{:}'.format(self.config['solver']['decode_lm_weight'])
        
        # Check models dev performance before inference
        self.asr_model.eval()
        self.asr_model.clear_att()
        self.asr_model = self.asr_model.to(self.device)
        self.verbose('Checking models performance on dev set '+str(self.config['solver']['dev_set'])+'...')
        self.valid()
        self.asr_model = self.asr_model.to('cpu') # move origin model to cpu, clone it to GPU for each thread

    def exec(self):
        '''Perform inference step with beam search decoding.'''
        test_cer = 0.0
        self.decode_beam_size = self.config['solver']['decode_beam_size']
        self.verbose('Start decoding with beam search, beam size = '+str(self.config['solver']['decode_beam_size']))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set),self.njobs))
        ## self.test_set = [(x,y) for (x,y) in self.test_set][::10]
        _ = Parallel(n_jobs=self.njobs)(delayed(self.beam_decode)(x[0],y[0].tolist()[0]) for x,y in tqdm(self.test_set))
        
        self.verbose('Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
        
        self.verbose('Top {} results at {}.'.format(self.config['solver']['decode_beam_size'],
                                                    str(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'))))
        
    def write_hyp(self,hyps,y):
        '''Record decoding results'''
        gt = self.mapper.translate(y,return_string=True)
        # Best
        with open(os.path.join(self.ckpdir,self.decode_file+'.txt'),'a') as f:
            best_hyp = self.mapper.translate(hyps[0].outIndex,return_string=True)
            f.write(gt+'\t'+best_hyp+'\n')
        # N best
        with open(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'),'a') as f:
            for hyp in hyps:
                best_hyp = self.mapper.translate(hyp.outIndex,return_string=True)
                f.write(gt+'\t'+best_hyp+'\n')
        

    def beam_decode(self,x,y):
        '''Perform beam decoding with end-to-end ASR'''
        # Prepare data
        x = x.to(device = self.device,dtype=torch.float32)
        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]

        # Forward
        with torch.no_grad():
            max_decode_step =  int(np.ceil(state_len[0]*self.decode_step_ratio))
            model = copy.deepcopy(self.asr_model).to(self.device)
            hyps = model.beam_decode(x, max_decode_step, state_len, self.decode_beam_size)
        del model
        
        self.write_hyp(hyps,y)
        del hyps
        
        return 1

    
    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        val_cer = 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        ctc_results = []
        with torch.no_grad():
            for cur_b,(x,y) in enumerate(self.dev_set):
                self.progress(' '.join(['Valid step - (',str(cur_b),'/',str(len(self.dev_set)),')']))

                # Prepare data
                if len(x.shape)==4: x = x.squeeze(0)
                if len(y.shape)==3: y = y.squeeze(0)
                x = x.to(device = self.device,dtype=torch.float32)
                y = y.to(device = self.device,dtype=torch.long)
                state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # Forward
                ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)
                ctc_pred = torch.argmax(ctc_pred,dim=-1).cpu() if ctc_pred is not None else None
                ctc_results.append(ctc_pred)

                # Result
                label = y[:,1:ans_len+1].contiguous()
                t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
                val_len += int(x.shape[0])
        
        
        # Dump model score to ensure model is corrected
        self.verbose('Validation Error Rate of Current model : {:.4f}      '.format(val_cer/val_len)) 
        self.verbose('See {} for validation results.'.format(os.path.join(self.ckpdir,'dev_att_decode.txt'))) 
        with open(os.path.join(self.ckpdir,'dev_att_decode.txt'),'w') as f:
            for hyp,gt in zip(all_pred,all_true):
                f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')
        
        # Also dump CTC result if available
        if ctc_results[0] is not None:
            ctc_results = [i for ins in ctc_results for i in ins]
            ctc_text = []
            for pred in ctc_results:
                p = [i for i in pred.tolist() if i != 0]
                p = [k for k, g in itertools.groupby(p)]
                ctc_text.append(self.mapper.translate(p,return_string=True))
            self.verbose('Also, see {} for CTC validation results.'.format(os.path.join(self.ckpdir,'dev_ctc_decode.txt'))) 
            with open(os.path.join(self.ckpdir,'dev_ctc_decode.txt'),'w') as f:
                for hyp,gt in zip(ctc_text,all_true):
                    f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')


class RNNLM_Trainer(Solver):
    ''' Trainer for RNN-LM only'''
    def __init__(self, config, paras):
        super(RNNLM_Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_dev_ppx = 1000

        # training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        self.apex = config['solver']['apex']

    def load_data(self):
        ''' Load training / dev set'''
        self.verbose('Loading text data from '+self.config['solver']['data_path'])
        setattr(self,'train_set', get_Dataloader('train', load='text', use_gpu=self.paras.gpu, **self.config['solver']))
        setattr(self,'dev_set', get_Dataloader('dev', load='text', use_gpu=self.paras.gpu, **self.config['solver']))

    def set_model(self):
        ''' Setup RNNLM'''
        self.verbose('Init RNNLM model.')
        self.rnnlm = RNN_LM(out_dim=self.mapper.get_dim(),**self.config['rnn_lm']['model_para'])
        self.rnnlm = self.rnnlm.to(self.device)

        if self.paras.load:
            raise NotImplementedError

        # optimizer
        if self.apex and self.config['rnn_lm']['optimizer']['type']=='Adam':
            import apex
            self.rnnlm_opt = apex.optimizers.FusedAdam(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'])
        else:
            self.rnnlm_opt = getattr(torch.optim,self.config['rnn_lm']['optimizer']['type'])
            self.rnnlm_opt = self.rnnlm_opt(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'],eps=1e-8)

    def exec(self):
        ''' Training RNN-LM'''
        self.verbose('RNN-LM Training set total '+str(len(self.train_set))+' batches.')

        while self.step < self.max_step:
            for y in self.train_set:
                self.progress('Training step - '+str(self.step))
                # Load data
                if len(y.shape)==3: y = y.squeeze(0)
                y = y.to(device = self.device,dtype=torch.long)
                ans_len = torch.sum(y!=0,dim=-1)

                self.rnnlm_opt.zero_grad()
                _, prob = self.rnnlm(y[:,:-1],ans_len)
                loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
                loss.backward()
                self.rnnlm_opt.step()

                # logger
                ppx = torch.exp(loss.cpu()).item()
                self.log.add_scalars('perplexity',{'train':ppx},self.step)

                # Next step
                self.step += 1
                if self.step % self.valid_step ==0:
                    self.valid()
                if self.step > self.max_step:
                    break

    def valid(self):
        self.rnnlm.eval()

        print_loss = 0.0
        dev_size = 0 

        for cur_b,y in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(self.step),'(',str(cur_b),'/',str(len(self.dev_set)),')']))
            if len(y.shape)==3: y = y.squeeze(0)
            y = y.to(device = self.device,dtype=torch.long)
            ans_len = torch.sum(y!=0,dim=-1)
            _, prob = self.rnnlm(y[:,:-1],ans_len)
            loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
            print_loss += loss.clone().detach() * y.shape[0]
            dev_size += y.shape[0]

        print_loss /= dev_size
        dev_ppx = torch.exp(print_loss).cpu().item()
        self.log.add_scalars('perplexity',{'dev':dev_ppx},self.step)
        
        # Store model with the best perplexity
        if dev_ppx < self.best_dev_ppx:
            self.best_dev_ppx  = dev_ppx
            self.verbose('Best val ppx      : {:.4f}       @ step {}'.format(self.best_dev_ppx,self.step))
            torch.save(self.rnnlm,os.path.join(self.ckpdir,'rnnlm'))

        self.rnnlm.train()
