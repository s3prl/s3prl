from __future__ import annotations

import torch.nn as nn

from s3prl.corpus.kaldi import kaldi_for_multiclass_tagging
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging
from s3prl.dataset.chunking import UnfoldChunkByFrame
from s3prl.dataset.common_pipes import LoadAudio, RenameItems
from s3prl.dataset.base import SequentialDataPipe
from s3prl.nn.rnn import SuperbSDModel
from s3prl.sampler import (
    FixedBatchSizeBatchSampler,
    MaxTimestampBatchSampler,
    GroupSameItemSampler,
)
from s3prl.task.diar import DiarizationPIT
from s3prl.util.configuration import override_parent_cfg

from .base import SuperbProblem


class SuperbSDDatapipe(SequentialDataPipe):
    def __init__(
        self,
        upstream_rate: int,
        sample_rate: int = 16000,
        **kwds,
    ):
        super().__init__(
            UnfoldChunkByFrame(
                min_chunk_frames=2000,
                max_chunk_frames=2000,
                step_frames=2000,
                feat_frame_shift=upstream_rate,
                sample_rate=sample_rate,
            ),
            BuildMultiClassTagging(
                sample_rate=sample_rate, feat_frame_shift=upstream_rate
            ),
            LoadAudio(crop_segment=True, audio_sample_rate=sample_rate),
            RenameItems(
                x="wav",
                x_len="wav_len",
                label="multiclass_tag",
                rec_id="unchunked_id",
                order_in_rec="chunk_index",
            ),
        )


class SuperbSD(SuperbProblem):
    """
    Superb Intent Classification problem
    """

    @override_parent_cfg(
        corpus=dict(
            _cls=kaldi_for_multiclass_tagging,
            dataset_root="???",
        ),
        train_datapipe=dict(
            _cls=SuperbSDDatapipe,
        ),
        train_sampler=dict(
            _cls=MaxTimestampBatchSampler,
            max_timestamp=16000 * 200,
            shuffle=True,
        ),
        valid_datapipe=dict(
            _cls=SuperbSDDatapipe,
        ),
        valid_sampler=dict(
            _cls=FixedBatchSizeBatchSampler,
            batch_size=2,
        ),
        test_datapipe=dict(
            _cls=SuperbSDDatapipe,
        ),
        test_sampler=dict(
            _cls=GroupSameItemSampler,
            item_name="rec_id",
            item_order_name="order_in_rec",
        ),
        downstream=dict(
            _cls=SuperbSDModel,
            output_size=2,  # speaker num per recording
            hidden_size=256,
            rnn_layers=1,
        ),
        task=dict(
            _cls=DiarizationPIT,
        ),
    )
    @classmethod
    def setup_problem(cls, **cfg):
        """
        This setups the IC problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup_problem(**cfg)

    @override_parent_cfg(
        optimizer=dict(
            _cls="torch.optim.Adam",
            lr=1.0e-4,
        ),
        trainer=dict(
            total_steps=1000,
            log_step=100,
            eval_step=500,
            save_step=100,
            gradient_clipping=1.0,
            gradient_accumulate_steps=4,
            valid_metric="der",
            valid_higher_better=False,
        ),
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @override_parent_cfg()
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @override_parent_cfg(
        start_stage=0,
        final_stage=2,
        stage_0=dict(
            _method="setup_problem",
        ),
        stage_1=dict(
            _method="train",
        ),
        stage_2=dict(
            _method="inference",
        ),
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)
