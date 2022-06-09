from __future__ import annotations

import re
import logging
from pathlib import Path
import subprocess
import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt

from s3prl.base import Container
from s3prl.base.workspace import Workspace
from s3prl.corpus.kaldi import kaldi_for_multiclass_tagging
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging
from s3prl.dataset.chunking import UnfoldChunkByFrame
from s3prl.dataset.common_pipes import LoadAudio, RenameItems
from s3prl.dataset.base import SequentialDataPipe
from s3prl.encoder.category import CategoryEncoder
from s3prl.nn.rnn import SuperbSDModel
from s3prl.sampler import (
    FixedBatchSizeBatchSampler,
    MaxTimestampBatchSampler,
    GroupSameItemSampler,
)
from s3prl.task.diarization import DiarizationPIT
from s3prl.util.configuration import default_cfg, override_parent_cfg, field

from .base import SuperbProblem

logger = logging.getLogger(__name__)


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


def prediction_numpy_to_segment_secs(
    prediction: np.ndarray,
    threshold: float = 0.5,
    median_filter: int = 1,
    frame_shift: int = 160,
    subsampling: int = 1,
    sampling_rate: int = 16000,
    tag_category: CategoryEncoder = None,
):
    """
    prediction: (timestamps, class_num), all values are in 0~1
    """
    hard_pred = np.where(prediction > threshold, 1, 0)
    if median_filter > 1:
        hard_pred = medfilt(hard_pred, (median_filter, 1))
    factor = frame_shift * subsampling / sampling_rate
    segments = dict()
    for classid, frames in enumerate(hard_pred.T):
        frames = np.pad(frames, (1, 1), "constant")
        (changes,) = np.where(np.diff(frames, axis=0) != 0)
        if len(changes) > 0:
            if tag_category is not None:
                class_name = tag_category.decode(classid)
            else:
                class_name = str(classid)
            segments[class_name] = []
            for s, e in zip(changes[::2], changes[1::2]):
                start = s * factor
                end = e * factor
                segments[class_name].append((start, end))
    return segments


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

    @default_cfg(
        workspace="???",
        prediction=field(
            "prediction",
            "The directory name under the workspace containing all the predicted numpy",
        ),
        test_data=field("test_data", "The testing data (in dict) under this workspace"),
        median_filters=field([1, 11], "The median filter sizes to try when scoring"),
        thresholds=field(
            [0.3, 0.4, 0.5, 0.6, 0.7],
            "The threshold to try when determining 0/1 hard prediction.\n"
            "The raw predictions are all between 0~1\n",
        ),
        frame_shift=field(
            None,
            "The frame shift of the prediction np.ndarray. Used to map the frame-level prediction back to seconds",
        ),
    )
    @classmethod
    def scoring(cls, **cfg):
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)
        frame_shift = cfg.frame_shift or workspace.environ["frame_shift"]
        test_data: dict = workspace[cfg.test_data]
        test_segments = {
            reco: data_point["segments"] for reco, data_point in test_data.items()
        }
        test_rttm = workspace.put(test_segments, "test_rttm", "rttm")

        rttm_dir = workspace / "rttm"
        scoring_dir = workspace / "scoring"
        scoring_dir.mkdir(exist_ok=True, parents=True)
        all_ders = []
        for median_filter in cfg.median_filters:
            for threshold in cfg.thresholds:
                logger.info(
                    "Decode prediction numpy array with the setting: median filter="
                    f"{median_filter}, threshold={threshold}"
                )
                all_segments = dict()
                workspace = Workspace(workspace)
                at_least_one_segment = False
                for p in tqdm(
                    (workspace / cfg.prediction).files(), desc="prediction to seconds"
                ):
                    segments = prediction_numpy_to_segment_secs(
                        (workspace / cfg.prediction)[p],
                        threshold,
                        median_filter,
                        frame_shift,
                    )
                    if len(segments) > 0:
                        at_least_one_segment = True
                    all_segments[p] = segments
                if not at_least_one_segment:
                    logger.info("No segments found under this decoding setting")
                    continue
                identifier = f"hyp_threshold-{threshold}_median-{median_filter}"
                hyp_rttm = rttm_dir.put(all_segments, identifier, "rttm")
                overall_der = cls.score_with_dscore(
                    dscore_dir=workspace / "dscore",
                    hyp_rttm=hyp_rttm,
                    gt_rttm=test_rttm,
                    score_file=Path(scoring_dir / identifier),
                )
                logger.info(
                    f"Overall DER with median_filter {median_filter} and threshold {threshold}: {overall_der}"
                )
                all_ders.append(overall_der)
        all_ders.sort()
        best_der = all_ders[0]
        logger.info(f"Best DER on test data: {best_der}")
        workspace.put(dict(der=best_der), "test_metric", "yaml")

    @override_parent_cfg(
        start_stage=0,
        final_stage=3,
        stage_0=dict(
            _method="setup_problem",
        ),
        stage_1=dict(
            _method="train",
        ),
        stage_2=dict(
            _method="inference",
        ),
        stage_3=dict(
            _method="scoring",
        ),
    )
    @classmethod
    def run_stages(cls, **cfg):
        super().run_stages(**cfg)

    @default_cfg(
        dscore_dir=field("???", "The directory containing the 'dscore' repository"),
        hyp_rttm=field("???", "The hypothesis rttm file"),
        gt_rttm=field("???", "The ground truth rttm file"),
        score_file=field("???", "The scored result file"),
    )
    @classmethod
    def score_with_dscore(cls, **cfg) -> float:
        """
        This function returns the overall DER score, and will also write the detailed scoring results
        to 'score_file'
        """
        cfg = Container(cfg)
        dscore_dir = Workspace(cfg.dscore_dir)
        if not dscore_dir.is_dir() or "score" not in dscore_dir.files():
            subprocess.check_output(
                f"git clone https://github.com/nryant/dscore.git {dscore_dir}",
                shell=True,
            ).decode("utf-8")
        result = subprocess.check_call(
            f"python3 {dscore_dir}/score.py -s {cfg.gt_rttm} -r {cfg.hyp_rttm} > {cfg.score_file}",
            shell=True,
        )
        assert result == 0, "The scoring step fail."
        with open(cfg.score_file) as file:
            lines = file.readlines()
            overall_line = lines[-2].strip()
            overall_line = re.sub(" +", " ", overall_line)
            overall_line = re.sub("\t+", " ", overall_line)
            overall_der = float(overall_line.split(" ")[3])
        return overall_der
