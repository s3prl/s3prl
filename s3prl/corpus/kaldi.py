# -*- coding: utf-8 -*- #
"""
Author       Jiatong Shi, Leo Yang
Source       Refactored from https://github.com/hitachi-speech/EEND
Copyright    Copyright(c), Johns Hopkins University, National Taiwan University
"""

import os
from pathlib import Path
from collections import defaultdict

from s3prl import Container


def _kaldi_for_multiclass_tagging(split_root: str):
    datadir = KaldiDataDir(split_root)
    data = dict()
    for reco, path in datadir.wavs.items():
        data[reco] = dict()
        data[reco]["wav_path"] = path
        data[reco]["start_sec"] = 0
        data[reco]["end_sec"] = datadir.reco2dur[reco]
        data[reco]["segments"] = defaultdict(list)

    for utt, (reco, st, et) in datadir.segments.items():
        spk = datadir.utt2spk[utt]
        data[reco]["segments"][spk].append((st, et))

    return data


def kaldi_for_multiclass_tagging(dataset_root: str, splits=["train", "dev", "test"]):
    splits_data = []
    for split in splits:
        splits_data.append(_kaldi_for_multiclass_tagging(Path(dataset_root) / split))
    return Container(
        train=splits_data[0],
        valid=splits_data[1],
        test=splits_data[2],
    )


class KaldiDataDir:
    """This class holds data in kaldi-style directory."""

    def __init__(self, dataset_root: str):
        """Load kaldi data directory."""
        self.data_dir = dataset_root
        self.segments = self._load_segments(os.path.join(self.data_dir, "segments"))
        self.utt2spk = self._load_utt2spk(os.path.join(self.data_dir, "utt2spk"))
        self.wavs = self._load_wav_scp(os.path.join(self.data_dir, "wav.scp"))
        self.reco2dur = self._load_reco2dur(os.path.join(self.data_dir, "reco2dur"))
        self.spk2utt = self._load_spk2utt(os.path.join(self.data_dir, "spk2utt"))

    def _load_segments(self, segments_file):
        """Load segments file as dict with uttid index."""
        ret = {}
        if not os.path.exists(segments_file):
            return None
        for line in open(segments_file):
            utt, rec, st, et = line.strip().split()
            ret[utt] = (rec, float(st), float(et))
        return ret

    def _load_wav_scp(self, wav_scp_file):
        """Return dictionary { rec: wav_rxfilename }."""
        lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
        return {x[0]: x[1] for x in lines}

    def _load_utt2spk(self, utt2spk_file):
        """Returns dictionary { uttid: spkid }."""
        lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
        return {x[0]: x[1] for x in lines}

    def _load_spk2utt(self, spk2utt_file):
        """Returns dictionary { spkid: list of uttids }."""
        if not os.path.exists(spk2utt_file):
            return None
        lines = [line.strip().split() for line in open(spk2utt_file)]
        return {x[0]: x[1:] for x in lines}

    def _load_reco2dur(self, reco2dur_file):
        """Returns dictionary { recid: duration }."""
        if not os.path.exists(reco2dur_file):
            return None
        lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
        return {x[0]: float(x[1]) for x in lines}
