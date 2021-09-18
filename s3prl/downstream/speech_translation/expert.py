import os
import math
import torch
import random

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

import sentencepiece
import sacrebleu
from tqdm.auto import tqdm
import editdistance
import fairseq
from argparse import Namespace
from .S3prl_SpeechToTextTask import S3prl_SpeechToTextTask
from .AdditionalDataset import AdditionalDataset
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable, S2TTransformerModel
from fairseq.models.transformer import Embedding
from fairseq.data import Dictionary, encoders
import string

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.
                
                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()

        print(downstream_expert)

        self.expdir = expdir
        self.src_lang = downstream_expert['src_lang']
        self.tgt_lang = downstream_expert['tgt_lang']
        self.post_process = downstream_expert['post_process']
        self.output_prefix = downstream_expert['output_prefix']
        self.upstream_rate = upstream_rate

        self.datarc = downstream_expert['datarc']
        self.max_positions = downstream_expert['modelrc']['max_source_positions']

        self.upstream_rate = downstream_expert.get('upstream_rate', upstream_rate)
        if self.upstream_rate < 0:
            self.upstream_rate = upstream_rate
        assert self.upstream_rate % upstream_rate == 0
        self.downsample_ratio = int(self.upstream_rate / upstream_rate)
        self.downsample_method = downstream_expert.get('downsample_method', 'drop')
        if self.downsample_method == 'concat':
            upstream_dim *= self.downsample_ratio

        self.task = S3prl_SpeechToTextTask.setup_task(Namespace(**downstream_expert['taskrc']))
        self.task.upstream_rate = self.upstream_rate

        self.data_dir = downstream_expert['taskrc']['data']

        self.criterion = self.task.build_criterion(Namespace(**downstream_expert['criterionrc']))

        modelrc = Namespace(**downstream_expert['modelrc'])
        assert modelrc.arch in fairseq.models.ARCH_CONFIG_REGISTRY
        fairseq.models.ARCH_CONFIG_REGISTRY[modelrc.arch](modelrc)
        self.model = self.task.build_model(modelrc, upstream_dim)

        self.generator = self.task.build_generator([self.model], Namespace(**downstream_expert['generatorrc']))
        self.batch_itr = {}
        
        self.use_asr = downstream_expert['taskrc']['use_asr']

        if self.use_asr:

            rc = downstream_expert['asrrc']
            self.asr_datarc = rc['datarc']
            self.asr_weight = rc['weight']
            self.asr_dict = Dictionary.load(f"{self.data_dir}/{rc['vocab_file']}")
            
            asr_bperc = rc['bpe_tokenizer'].copy()
            asr_bperc['sentencepiece_model'] = f"{self.data_dir}/{asr_bperc['sentencepiece_model']}" 
            self.asr_bpe = encoders.build_bpe(Namespace(**asr_bperc))
            
            self.asr_task = S3prl_SpeechToTextTask.setup_task(Namespace(**downstream_expert['taskrc']))
            self.asr_dict.add_symbol('<blank>')
            self.asr_task.tgt_dict = self.asr_dict
            self.asr_head = nn.Linear(modelrc.encoder_embed_dim, len(self.asr_dict))
            self.additional_dataset = {}

        self.register_buffer('best_score', torch.zeros(1))

    # Interface
    def get_dataloader(self, split, epoch: int = 0):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """


        data_split = self.datarc[split]

        # load dataset
        if data_split not in self.batch_itr:
            
            # dataset will truncate the input wav according to model's max_position
            self.task.load_dataset(split=data_split, max_feature_len=self.max_positions)
            
            # it must not have invalid_inputs due to truncation
            self.batch_itr[data_split] = self.task.get_batch_iterator(
                self.task.dataset(data_split),
                max_tokens=self.datarc['max_tokens'],
                max_positions=self.max_positions,
                num_workers=self.datarc['num_workers'],
                ignore_invalid_inputs = False,
                epoch=epoch+1,
            )

        # # fairseq's dataloader
        # # note: refreshing is needed for each epoch
        return self.batch_itr[data_split].next_epoch_itr()

    # Interface
    def forward(self, mode, features, input_dict, records, **kwargs):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        device = features[0].device
        features = self.downsample(features)
        features_length = torch.LongTensor([len(feature) for feature in features])
        features = pad_sequence(features, batch_first=True, padding_value=0.0)

        input_dict['net_input']['src_tokens'] = features
        input_dict['net_input']['src_lengths'] = features_length

        input_dict = fairseq.utils.move_to_cuda(input_dict, device=device)

        if self.use_asr:
            asr_input_dict = self._create_asr_input_dict(input_dict, mode)
            asr_input_dict = fairseq.utils.move_to_cuda(asr_input_dict, device=device)

        loss = torch.FloatTensor(0)

        if mode in ['train', 'dev']:

            encoder_out = self.model.encoder(
                src_tokens=input_dict['net_input']['src_tokens'], src_lengths=input_dict['net_input']['src_lengths']
            )

            st_decoder_out = self.model.decoder(
                prev_output_tokens=input_dict['net_input']['prev_output_tokens'],
                encoder_out=encoder_out
            )

            st_loss, _ = self.criterion.compute_loss(
                self.model,
                st_decoder_out,
                input_dict,
            )

            loss = st_loss

            if self.use_asr:

                asr_loss = self.count_asr_loss(encoder_out, asr_input_dict)
                loss = (1-self.asr_weight) * st_loss + self.asr_weight * asr_loss

            loss /= input_dict['nsentences']

            records['loss'].append(loss.item())

            if self.use_asr:

                records['st_loss'].append(st_loss.item())
                records['asr_loss'].append(asr_loss.item())


        if mode in ['dev', 'test']:

            records['ids'] += input_dict['id'].cpu().tolist()
            records['utt_ids'] += input_dict['utt_id']

            hyps, refs = self._inference_step(input_dict)
            records['hyps'] += hyps
            records['refs'] += refs

            if self.use_asr:
                asr_hyps, asr_refs = self._inference_step_asr(asr_input_dict)
                records['asr_hyps'] += asr_hyps
                records['asr_refs'] += asr_refs

        return loss

    def downsample(self, features):

        if self.downsample_ratio == 1:
            return features
        
        new_features = []
        
        for feature in features:
            if self.downsample_method == 'drop':

                feature = feature[::self.downsample_ratio]
            
            elif self.downsample_method == 'concat':
                
                N = feature.size(0) % self.downsample_ratio
                if N != 0:
                    feature = F.pad(feature, (0, 0, 0, self.downsample_ratio-N))
                feature = feature.view(feature.size(0)//self.downsample_ratio, feature.size(1)*self.downsample_ratio)
            
            elif self.downsample_method == 'average':

                N = feature.size(0) % self.downsample_ratio
                if N != 0:
                    feature = F.pad(feature, (0, 0, 0, self.downsample_ratio-N))
                feature = feature.view(feature.size(0)//self.downsample_ratio, self.downsample_ratio, feature.size(1)).mean(dim=1)

            else:
                raise NotImplementedError
            
            new_features.append(feature)
        
        return new_features

    def _create_asr_input_dict(self, input_dict, mode):

        if mode not in self.additional_dataset:
            
            dataset = AdditionalDataset.from_tsv(
                f'{self.data_dir}/{self.datarc[mode]}.tsv',
                self.asr_datarc['key'],
                self.asr_dict,
                self.asr_bpe,
            )
            self.additional_dataset[mode] = dataset

        additional_data = self.additional_dataset[mode].get_addtional_input(input_dict['id'])

        asr_input_dict = input_dict.copy()
        asr_input_dict['net_input'] = input_dict['net_input'].copy()
        asr_input_dict['net_input']['prev_output_tokens'] = additional_data['prev_output_tokens']
        asr_input_dict['target'] = additional_data['target']
        asr_input_dict['target_lengths'] = additional_data['target_lengths']
        asr_input_dict['ntokens'] = additional_data['ntokens']

        return asr_input_dict

    def count_asr_loss(self, encoder_out, input_dict):

        hidden = encoder_out['encoder_out'][0] # T x B x C
        log_prob = self.asr_head(hidden).log_softmax(2)

        hidden_length = self.model.encoder.subsample.get_out_seq_lens_tensor(input_dict['net_input']['src_lengths'])
        
        targets = input_dict['target']
        target_lengths = input_dict['target_lengths']

        loss = nn.functional.ctc_loss(log_prob, targets, hidden_length, target_lengths, blank=self.asr_dict.index('<blank>'), reduction='sum', zero_infinity=True)

        return loss

    def _decode(self, toks, dictionary):

        toks = toks[toks != dictionary.pad()]

        s = dictionary.string(
            toks.int().cpu(),
            self.post_process,
        )

        return s if s else "<unk>"

    def _inference_step(self, input_dict):
        output = self.generator.generate([self.model], input_dict)

        hyps = []
        refs = []

        for i in range(len(output)):

            hyps.append(
                self._decode(output[i][0]["tokens"], self.task.target_dictionary)
            )

            refs.append(
                self._decode(input_dict['target'][i], self.task.target_dictionary)
            )

        return hyps, refs

    def _inference_step_asr(self, input_dict):
        
        encoder_out = self.model.encoder(
            src_tokens=input_dict['net_input']['src_tokens'], src_lengths=input_dict['net_input']['src_lengths']
        )

        hidden = encoder_out['encoder_out'][0] # TxBxC
        logit = self.asr_head(hidden)

        predict = logit.argmax(dim=-1).transpose(0, 1)
        
        hyps = []
        refs = []

        for i in range(len(predict)):

            predict_ids = predict[i].unique_consecutive()
            predict_ids = predict_ids[predict_ids != self.asr_dict.index('<blank>')]

            hyps.append(
                self._decode(predict_ids, self.asr_dict)
            )

            refs.append(
                self._decode(input_dict['target'][i], self.asr_dict)
            )
        
        return hyps, refs

    def _metric(self, hyps, refs):

        tok = 'zh' if self.tgt_lang == 'zh' else '13a'
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)

        return bleu

    def _asr_metric(self, hyps, refs):

        ce = 0
        we = 0
        c_total = 0
        w_total = 0

        for hyp, ref in zip(hyps, refs):

            normalized_hyp = hyp.translate(str.maketrans('', '', "".join(list(set(string.punctuation)-set("'-"))))).lower()
            normalized_ref = ref.translate(str.maketrans('', '', "".join(list(set(string.punctuation)-set("'-"))))).lower()

            ce += editdistance.eval(normalized_hyp, normalized_ref)
            c_total += len(normalized_ref)

            hyp_w = normalized_hyp.split()
            ref_w = normalized_ref.split()

            we += editdistance.eval(hyp_w, ref_w)
            w_total += len(ref_w)


        cer = ce / c_total
        wer = we / w_total

        return cer, wer

    # interface
    def log_records(self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        """
        Args:
            mode: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev' or 'test' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{mode}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader
        
        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        save_names = []

        if mode in ['train', 'dev']:

            ave_loss = sum(records['loss'])/len(records['loss'])
            logger.add_scalar(
                f'st/{mode}-loss',
                ave_loss,
                global_step=global_step
            )

            if self.use_asr:

                ave_st_loss = sum(records['st_loss'])/len(records['st_loss'])
                logger.add_scalar(
                    f'st/{mode}-st_loss',
                    ave_st_loss,
                    global_step=global_step
                )

                ave_asr_loss = sum(records['asr_loss'])/len(records['asr_loss'])
                logger.add_scalar(
                    f'st/{mode}-asr_loss',
                    ave_asr_loss,
                    global_step=global_step
                )

        if mode in ['dev', 'test']:

            bleu = self._metric(records['hyps'], records['refs'])
            logger.add_scalar(
                f'st/{mode}-bleu',
                bleu.score,
                global_step=global_step
            )
            for i in range(4):
                logger.add_scalar(
                    f'st/{mode}-bleu{i+1}',
                    bleu.precisions[i],
                    global_step=global_step
                )

            if bleu.score > self.best_score and mode == 'dev':
                self.best_score = torch.ones(1) * bleu.score
                save_names.append(f'{mode}-best.ckpt') 
            
            with open(f'{self.expdir}/{self.output_prefix}-st-{mode}.tsv', 'w') as f:
                print('utt_id', 'hyp', 'ref', sep='\t', file=f)
                results = list(zip(records['ids'], records['hyps'], records['refs'], records['utt_ids']))
                results.sort(key=lambda x: x[0])
                for idx, hyp, ref, utt_id in results:
                    print(utt_id, hyp, ref, sep='\t', file=f)

            print(bleu)

            if self.use_asr:

                cer, wer = self._asr_metric(records['asr_hyps'], records['asr_refs'])
                logger.add_scalar(
                    f'st/{mode}-asr-cer',
                    cer,
                    global_step=global_step
                )
                logger.add_scalar(
                    f'st/{mode}-asr-wer',
                    wer,
                    global_step=global_step
                )

                with open(f'{self.expdir}/{self.output_prefix}-asr-{mode}.tsv', 'w') as f:
                    print('utt_id', 'hyp', 'ref', sep='\t', file=f)
                    results = list(zip(records['ids'], records['asr_hyps'], records['asr_refs'], records['utt_ids']))
                    results.sort(key=lambda x: x[0])
                    for idx, hyp, ref, utt_id in results:
                        print(utt_id, hyp, ref, sep='\t', file=f)

                tqdm.write(f'[cer]:{cer}, [wer]:{wer}')
        
        return save_names