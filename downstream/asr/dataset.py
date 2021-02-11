# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import logging
import os
import random
#-------------#
import pandas as pd
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
#-------------#
from dictionary import Dictionary


HALF_BATCHSIZE_TIME = 2000


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):
    
    def __init__(self, split, bucket_size, libri_root, bucket_file, sample_rate=16000, train_dev_seed=1337, **kwargs):
        super(SequenceDataset, self).__init__()
        
        self.libri_root = libri_root
        self.sample_rate = sample_rate

        self.split_sets = dict(
            train = ["train-clean-100", "train-clean-360", "train-other-500"],
            dev = ["dev-clean", "dev-other"],
            test = ["test-clean", "test-other"],
        )

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `preprocess/generate_len_for_bucket.py to get bucket file.'

        # Wavs
        if split in ["train", "dev", "test"]:
            table_list = []
            for item in self.split_sets[split]:
                file_path = os.path.join(bucket_file, item + ".csv")
                if os.path.exists(file_path):
                    table_list.append(
                        pd.read_csv(file_path)
                    )
                else:
                    logging.warning(f'{item} is not found in bucket_file: {bucket_file}, skipping it.')

            table_list = pd.concat(table_list)
            table_list = table_list.sort_values(by=['length'], ascending=False)
        else:
            raise ValueError('Invalid \'split\' argument for dataset: SequenceDataset!')

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()

        # Transcripts
        Y = self._load_transcript(X)

        x_names = set([self._parse_x_name(x) for x in X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)

        Y = {key: Y[key] for key in usage_list}

        # dictionary, symbol list
        self.dictionary = self._build_dictionary(Y)
        self.symbols = self.dictionary.symbols

        self.Y = {k: self.dictionary.encode_line(v).long() for k, v in Y.items()}

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)
                
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        # assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def _load_transcript(self, x_list):
        """Load the transcripts for Librispeech"""
        def process_trans(transcript):
            #TODO: support character / bpe
            transcript = transcript.upper()
            return " ".join(list(transcript.replace(" ", "|"))) + " |"

        trsp_sequences = {}
        split_spkr_chap_list = list(
            set(
                "/".join(x.split('/')[:-1]) for x in x_list
            )
        )

        for dir in split_spkr_chap_list:
            parts = dir.split('/')
            trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
            path = os.path.join(self.libri_root, dir, trans_path)
            assert os.path.exists(path)

            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))

        return trsp_sequences

    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(
            transcript_list, d, workers
        )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file) for x_file in self.X[index]]
        label_batch = [self.Y[self._parse_x_name(x_file)] for x_file in self.X[index]]
        return wav_batch, label_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
