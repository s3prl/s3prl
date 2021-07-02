from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.data import FairseqDataset, data_utils, iterators
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset, SpeechToTextDatasetCreator, S2TDataConfig
from typing import Dict, List, Optional, Tuple
import torchaudio
import torch
from argparse import Namespace
import os.path as op

class S3prl_SpeechToTextTask_TMP(SpeechToTextTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = S3prl_SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )
    
    def build_model(self, args, input_dim):
        # args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_feat_per_channel = input_dim
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        ):
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )
        # epoch_iter = torch.utils.data.DataLoader(
        #     dataset,
        #     batch_sampler=batch_sampler,
        #     num_workers=num_workers,
        #     collate_fn=dataset.collater,
        # )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

class S3prl_SpeechToTextDatasetCreator(SpeechToTextDatasetCreator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SpeechToTextDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        return S3prl_SpeechToTextDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )

class S3prl_SpeechToTextDataset(SpeechToTextDataset):

    SAMPLE_RATE = 48000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resampler = torchaudio.transforms.Resample(
            orig_freq = self.SAMPLE_RATE,
            new_freq = 16000,
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        # source = get_features_or_waveform(
        #     self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
        # )
        source, sr = torchaudio.load(self.audio_paths[index])
        source = self.resampler(source)
        source = source.view(-1)

        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = source.float()

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
        return index, source, target

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]):

        output_dict = super().collater(samples)

        wavs = []

        for i in range(output_dict['nsentences']):
            wav = output_dict['net_input']['src_tokens'][i]
            length = output_dict['net_input']['src_lengths'][i].item()
            wavs.append(wav[:length])

        return wavs, output_dict

class dummyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, batch_sampler):

        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def __getitem__(self, idx):
        batch = []
        for sample_id in self.batch_sampler[idx]:
            batch.append(self.dataset[sample_id])
        return self.dataset.collater(batch)

if __name__ == '__main__':
    args = Namespace(**{
        'task': 'speech_to_text',
        'data': '/livingrooms/public/CoVoST2/cv-corpus-6.1-2020-12-11/en/',
        'config_yaml': 'config_st_en_de.yaml',
        'max_tokens' : 40000,
        'criterion': 'label_smoothed_cross_entropy',
        'label_smoothing': 0.1,
        'arch': 's2t_transformer_xs',
        'seed': 1,
    })

    task = S3prl_SpeechToTextTask_TMP.setup_task(args)
    task.load_dataset('test_short')
    batch_itr = task.get_batch_iterator(
        task.dataset('test_short'), max_tokens=args.max_tokens,
    )
    dataset = dummyDataset(batch_itr.dataset, batch_itr.batch_sampler)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )
    for i in range(3):
        print(i)
        for batch in dataloader:
            print(batch[1]['id'])
    # batch_sampler = batch_itr.batch_sampler
    # print(batch_sampler)
    # print(type(batch_sampler))
    # for i in range(3):
    #     print(i)
    #     itr = batch_itr.next_epoch_itr()
    #     for batch in itr:
    #         print(batch[1]['id'])

