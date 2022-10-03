from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm

from s3prl.base import Container
from s3prl.base.workspace import Workspace
from s3prl.corpus.kaldi import kaldi_for_multiclass_tagging
from s3prl.dataset.base import SequentialDataPipe
from s3prl.dataset.chunking import UnfoldChunkByFrame
from s3prl.dataset.common_pipes import LoadAudio, SetOutputKeys
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging
from s3prl.encoder.category import CategoryEncoder
from s3prl.nn.rnn import SuperbDiarizationModel
from s3prl.sampler import FixedBatchSizeBatchSampler, GroupSameItemSampler
from s3prl.task.diarization import DiarizationPIT
from s3prl.util.configuration import default_cfg, field

from .base import SuperbProblem

logger = logging.getLogger(__name__)


class SuperbSDDatapipe(SequentialDataPipe):
    def __init__(
        self,
        feat_frame_shift: int,
        sample_rate: int = 16000,
        **kwds,
    ):
        super().__init__(
            UnfoldChunkByFrame(
                min_chunk_frames=2000,
                max_chunk_frames=2000,
                step_frames=2000,
                feat_frame_shift=feat_frame_shift,
                sample_rate=sample_rate,
            ),
            BuildMultiClassTagging(
                sample_rate=sample_rate, feat_frame_shift=feat_frame_shift
            ),
            LoadAudio(audio_sample_rate=sample_rate),
            SetOutputKeys(
                x="wav",
                x_len="wav_len",
                label="multiclass_tag",
                label_len="tag_len",
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

    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                CLS=kaldi_for_multiclass_tagging,
                dataset_root="???",
            ),
            train_datapipe=dict(
                CLS=SuperbSDDatapipe,
                train_category_encoder=True,
            ),
            train_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=8,
                shuffle=True,
            ),
            valid_datapipe=dict(
                CLS=SuperbSDDatapipe,
            ),
            valid_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            test_datapipe=dict(
                CLS=SuperbSDDatapipe,
            ),
            test_sampler=dict(
                CLS=GroupSameItemSampler,
                item_name="unchunked_id",
                item_order_name="chunk_index",
            ),
            downstream=dict(
                CLS=SuperbDiarizationModel,
                output_size=2,  # speaker num per recording
                hidden_size=512,
                rnn_layers=1,
            ),
            task=dict(
                CLS=DiarizationPIT,
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        """
        This setups the IC problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=30000,
                log_step=500,
                eval_step=500,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=4,
                valid_metric="der",
                valid_higher_better=False,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @default_cfg(**SuperbProblem.inference.default_cfg)
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
            int,
        ),
    )
    @classmethod
    def scoring(cls, **cfg):
        cfg = Container(cfg)
        workspace = Workspace(cfg.workspace)
        frame_shift = cfg.frame_shift or workspace.environ["feat_frame_shift"]
        test_data: dict = workspace[cfg.test_data]
        test_segments = {
            reco: data_point["segments"] for reco, data_point in test_data.items()
        }
        test_rttm = workspace.put(test_segments, "test_rttm", "rttm")

        rttm_dir = workspace / "rttm"
        scoring_dir = workspace / "scoring"
        scoring_dir.mkdir(exist_ok=True, parents=True)
        all_ders = []

        reco2pred = {}
        for p in tqdm((workspace / cfg.prediction).files(), desc="Load prediction"):
            reco2pred[p] = (workspace / cfg.prediction)[p]

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
                        reco2pred[p],
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

    @default_cfg(
        **SuperbProblem.run.default_except(
            stages=["setup", "train", "inference", "scoring"],
            start_stage="setup",
            final_stage="scoring",
            setup=setup.default_cfg.deselect("workspace", "resume", "dryrun"),
            train=train.default_cfg.deselect("workspace", "resume", "dryrun"),
            inference=inference.default_cfg.deselect("workspace", "resume", "dryrun"),
            scoring=scoring.default_cfg.deselect("workspace"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)

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
            f"python3 {dscore_dir}/score.py -r {cfg.gt_rttm} -s {cfg.hyp_rttm} > {cfg.score_file}",
            shell=True,
        )
        assert result == 0, "The scoring step fail."
        with open(cfg.score_file) as file:
            lines = file.readlines()
            overall_lines = [line for line in lines if "OVERALL" in line]
            assert len(overall_lines) == 1
            overall_line = overall_lines[0]
            overall_line = re.sub("\t+", " ", overall_line)
            overall_line = re.sub(" +", " ", overall_line)
            overall_der = float(overall_line.split(" ")[3])
            # The overall der line should look like:
            # *** OVERALL *** DER JER ...
        return overall_der
