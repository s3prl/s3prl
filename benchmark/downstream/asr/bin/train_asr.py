from collections import defaultdict

import os
import yaml
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from benchmark.downstream.asr.src.solver import BaseSolver

from benchmark.downstream.asr.src.asr import ASR
from benchmark.downstream.asr.src.optim import Optimizer
from benchmark.downstream.asr.src.data import load_dataset
from benchmark.downstream.asr.src.util import human_format, cal_er, feat_to_fig, LabelSmoothingLoss
from benchmark.downstream.asr.src.audio import Delta, Postprocess, Augment

EMPTY_CACHE_STEP = 100

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)

        self.val_mode = self.config['hparas']['val_mode'].lower()
        self.WER = 'per' if self.val_mode == 'per' else 'wer'

        self.upstream = torch.hub.load('andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning:benchmark', 'baseline_fbank')
        self.feat_dim = self.upstream.get_output_dim()
        self.load_data()
        self.set_model()

        self.register_buffer('best_er_att_dev', torch.ones(1) * 3.0)
        self.register_buffer('best_er_ctc_dev', torch.ones(1) * 3.0)
        self.register_buffer('best_er_att_test', torch.ones(1) * 3.0)
        self.register_buffer('best_er_ctc_test', torch.ones(1) * 3.0)


    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.vocab_size, self.tokenizer, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      False, **self.config['data'])
        self.verbose(msg)

        # Dev set sames
        self.dv_names = []
        if type(self.dv_set) is list:
            for ds in self.config['data']['corpus']['dev_split']:
                self.dv_names.append(ds[0])
        else:
            self.dv_names = self.config['data']['corpus']['dev_split'][0]


    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        #print(self.feat_dim) #160
        batch_size = self.config['data']['corpus']['batch_size']//2
        self.model = ASR(self.feat_dim, self.vocab_size, batch_size, **self.config['model'])



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

        # Scheduled sampling
        def get_teacher_forcing_scheduler(tf_start=1, tf_end=1, tf_step=1, tf_step_start=0, **kwargs):
            return lambda step: max(
                tf_end, 
                tf_start-(tf_start-tf_end)*(step-tf_step_start)/tf_step if step >= tf_step_start else 1
            )
        self.tf_scheduler = get_teacher_forcing_scheduler(**self.config['hparas'])

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.lr_scheduler = self.optimizer.lr_scheduler
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()
        
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

    
    def get_train_dataloader(self):
        return self.tr_set


    def get_dev_dataloader(self):
        return self.dv_set


    def get_test_dataloader(self):
        raise NotImplementedError


    def _forward_train(self, feat, feat_len, txt, txt_len, logger, global_step,
                       log_step, prefix='asr/train-', **kwargs):

        # Pre-step : update tf_rate/lr_rate and do zero_grad
        tf_rate = self.tf_scheduler(global_step)
        self.optimizer.pre_step(global_step)

        # Forward model
        # Note: txt should NOT start w/ <sos>
        ctc_output, encode_len, att_output, _, _ = \
            self.model(feat, feat_len, max(txt_len), tf_rate=tf_rate,
                       teacher=txt, get_dec_state=False)

        # Compute all objectives
        total_loss = 0
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
            att_loss = self.seq_loss(att_output.view(b*t,-1),txt.view(-1))
            # Sum each uttr and devide by length then mean over batch
            # att_loss = torch.mean(torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(txt!=0,dim=-1).float())
            total_loss += att_loss*(1-self.model.ctc_weight)

        if global_step % log_step == 0:
            if att_output is not None:
                logger.add_scalar(f'{prefix}att-loss', att_loss.item(), global_step=global_step)
                logger.add_scalar(f'{prefix}att-{self.WER}', cal_er(self.tokenizer,att_output,txt), global_step=global_step)
                logger.add_scalar(f'{prefix}att-cer', cal_er(self.tokenizer,att_output,txt,mode='cer'), global_step=global_step)
            if ctc_output is not None:
                logger.add_scalar(f'{prefix}ctc-loss', ctc_loss.item(), global_step=global_step)
                logger.add_scalar(f'{prefix}ctc-{self.WER}', cal_er(self.tokenizer,att_output,txt,ctc=True), global_step=global_step)
                logger.add_scalar(f'{prefix}ctc-cer', cal_er(self.tokenizer,att_output,txt,mode='cer',ctc=True), global_step=global_step)
                
                text = self.tokenizer.decode(ctc_output[0].argmax(dim=-1).tolist(), ignore_repeat=True)
                logger.add_text(f'{prefix}ctc-text', text, global_step=global_step)
        
        self.timer.cnt('fw')
        return total_loss


    def _forward_validate(self, feat, feat_len, txt, txt_len, logger, global_step,
                          records, prefix, batch_id, batch_num, **kwargs):
        # Forward model
        with torch.no_grad():
            ctc_output, encode_len, att_output, att_align, dec_state = \
                self.model(feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO))

        if att_output is not None:
            records['att_wer'].append(cal_er(self.tokenizer,att_output,txt,mode='wer'))
            records['att_cer'].append(cal_er(self.tokenizer,att_output,txt,mode='cer'))
            records['att_er'].append(cal_er(self.tokenizer,att_output,txt,mode=self.val_mode))
        if ctc_output is not None:
            records['ctc_wer'].append(cal_er(self.tokenizer,ctc_output,txt,mode='wer',ctc=True))
            records['ctc_cer'].append(cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True))
            records['ctc_er'].append(cal_er(self.tokenizer,ctc_output,txt,mode=self.val_mode,ctc=True))
        
        # Show some example on tensorboard
        if batch_id == batch_num // 2:
            for i in range(min(len(txt),self.DEV_N_EXAMPLE)):
                logger.add_text(f'{prefix}true-text-{i}', self.tokenizer.decode(txt[i].tolist()), global_step=global_step)
                if att_output is not None:
                    img, form = feat_to_fig(att_align[i,0,:,:].cpu().detach())
                    logger.add_image(f'{prefix}att-align-{i}', img, global_step=global_step, dataformats=form)
                    logger.add_text(f'{prefix}att-text-{i}', self.tokenizer.decode(att_output[i].argmax(dim=-1).tolist()), global_step=global_step)
                if ctc_output is not None:
                    logger.add_text(f'{prefix}ctc-text-{i}', self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(), ignore_repeat=True), global_step=global_step)

        return feat.new_zeros(1)


    def forward(self, feat, txt, *args, **kwargs):
        feat = pad_sequence(feat, batch_first=True)
        feat_len = torch.LongTensor([len(f) for f in feat]).to(feat.device)
        txt = txt.to(feat.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        self.timer.cnt('rd')

        if self.training:
            return self._forward_train(feat, feat_len, txt, txt_len, **kwargs)
        else:
            return self._forward_validate(feat, feat_len, txt, txt_len, **kwargs)


    def log_records(self, records, logger, prefix, global_step, **kwargs):
        if not self.training:
            tasks = []
            if 'att_er' in records:
                tasks.append('att')
            if 'ctc_er' in records:
                tasks.append('ctc')

            split = prefix.split('/')[-1][:-1]
            save_paths = []
            for task in tasks:
                def get_average(numbers):
                    return torch.FloatTensor(numbers).mean().item()

                avg_er = get_average(records[f'{task}_er'])
                avg_wer = get_average(records[f'{task}_wer'])
                avg_cer = get_average(records[f'{task}_cer'])

                buffer = getattr(self, f'best_er_{task}_{split}')
                if avg_er < buffer:
                    buffer.fill_(avg_er)
                    save_paths.append(f'best_{task}_{split}.ckpt')

                if global_step >= self.max_step:
                    save_paths.append(f'last_{task}_{split}.ckpt')

                logger.add_scalar(f'asr/{split}-{task}-{self.WER}'.lower(), avg_wer, global_step=global_step)
                logger.add_scalar(f'asr/{split}-{task}-cer'.lower(), avg_cer, global_step=global_step)

            return save_paths


    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        self.timer.set()
        
        records = defaultdict(list)
        while self.step< self.max_step:
            for batch_id, data in enumerate(self.tr_set):
                wavs, txt, _ = data
                feat = self.upstream([wav.to(self.device) for wav in wavs])

                total_loss = self.forward(feat, txt, global_step=self.step, logger=self.log, prefix='asr/train-',
                                          batch_id=batch_id, batch_num=len(self.tr_set), log_step=self.PROGRESS_STEP)
                grad_norm = self.backward(total_loss)

                self.step+=1

                # Validation
                if self.step % self.valid_step == 0:
                    if type(self.dv_set) is list:
                        for dv_id in range(len(self.dv_set)):
                            self.validate(self.dv_set[dv_id], self.dv_names[dv_id])
                    else:
                        self.validate(self.dv_set, self.dv_names)
                
                # Lr scheduling
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

                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step: break            
            
        self.log.close()
        print('[INFO] Finished training after', human_format(self.max_step), 'steps.')


    def validate(self, _dv_set, _name):
        print(f'Evaluating {_name} set.')

        # Eval mode
        self.eval()
        self.model.eval()

        records = defaultdict(list)
        for i,data in enumerate(_dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(_dv_set)))

            wavs, txt, _ = data
            feat = self.upstream([wav.to(self.device) for wav in wavs])

            self.forward(feat, txt, global_step=self.step, records=records, logger=self.log, prefix=f'asr/{_name}-',
                         batch_id=i, batch_num=len(_dv_set))

        filepaths = self.log_records(records=records, logger=self.log, prefix=f'asr/{_name}-', global_step=self.step)
        for filepath in filepaths:
            torch.save(self.state_dict(), os.path.join(self.ckpdir, filepath))

        # Resume training
        self.model.train()
        self.train()
