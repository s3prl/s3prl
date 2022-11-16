"""
The setting to pretrain TERA

Author
  * Andy T. Liu 2022
"""

import logging
import pickle
from pathlib import Path
from typing import List

import pandas as pd
import torch
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.dataio.corpus.librispeech import LibriSpeech
from s3prl.dataio.dataset import get_info
from s3prl.dataio.encoder.category import CategoryEncoders
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.dataset.pretrain_tera_pipe import PretrainTeraPipe
from s3prl.nn.predictor_mockingjay import PredictorMockingjay
from s3prl.nn.transformer_mockingjay import TransformerMockingjay
from s3prl.nn.upstream import UpstreamPredictorModel
from s3prl.task.feat_reconstruction_task import FeatReconstructionTask

from .run import Common

logger = logging.getLogger(__name__)


__all__ = [
    "libri_for_pretrain",
    "PretrainTera",
]


_input_size = 80
_mask_args = dict(
    position_encoding_size=768,  # int, this should be identical to `hidden_size`
    mask_proportion=0.15,  # float, mask this percentage of all spectrogram frames in each sequence at random during MAM training
    mask_consecutive_min=7,  # int, mask this amount of consecutive frames
    mask_consecutive_max=7,  # int, mask this amount of consecutive frames
    mask_allow_overlap=True,  # bool, allow overlap masking
    mask_bucket_ratio=1.5,  # float, only used when overlap is not allowed. sample a mask from each bucket in size of [sampled mask_consecutive * mask_bucket_ratio]
    mask_frequency=0.2,  # float, mask maximum this percentage of frequency bands, set to 0 for no frequency mask
)
_noise_args = dict(
    noise_proportion=0.0,  # float, for this percentage of the time, Gaussian noise will be applied on all frames during MAM training, set to 0 for no noise
)
_audio_config = dict(
    win_ms=25,
    hop_ms=10,
    n_freq=201,
    n_mels=_input_size,
    n_mfcc=13,
    input={
        "channel": 0,
        "cmvn": True,
        "delta": 0,
        "feat_type": "mel",
        "log": True,
    },
    target={
        "channel": 1,
        "cmvn": True,
        "delta": 0,
        "feat_type": "mel",
        "log": True,
    },
)
_transformer_config = dict(
    hidden_size=768,  # Size of the encoder layers and the pooler layer.
    num_hidden_layers=3,  # Number of hidden layers in the Transformer encoder.
    num_attention_heads=12,  # Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size=3072,  # The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act="gelu",  # The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
    hidden_dropout_prob=0.1,  # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities.
    initializer_range=0.02,  # The sttdev of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps=1.0e-12,  # The epsilon used by LayerNorm.
    share_layer=False,  # Share layer weights
    pre_layer_norm=False,  # To apply the pre layer normalization technique introduced in: https://arxiv.org/abs/2002.04745
)


def libri_for_pretrain(
    target_dir,
    cache_dir,
    dataset_root,
    train_sets: List[str],
    valid_sets: List[str],
    test_sets: List[str],
    n_jobs: int = 6,
    get_path_only: bool = False,
):
    """
    Prepare LibriSpeech for pretrain following :obj:`PretrainTera.prepare_data` format.
    See :obj:`LibriSpeech` for the arguments usage
    """
    target_dir = Path(target_dir)

    train_path = target_dir / f"{'+'.join(train_sets)}.csv"
    valid_path = target_dir / f"{'+'.join(valid_sets)}.csv"
    test_paths = [target_dir / f"{test_set}.csv" for test_set in test_sets]

    if get_path_only:
        return train_path, valid_path, test_paths

    corpus = LibriSpeech(dataset_root, n_jobs, train_sets, valid_sets, test_sets)
    train_data, valid_data, test_data = corpus.data_split

    def dict_to_csv(data_dict, csv_path):
        keys = sorted(list(data_dict.keys()))
        fields = sorted(data_dict[keys[0]].keys())
        data = dict()
        for field in fields:
            data[field] = []
            for key in keys:
                data[field].append(data_dict[key][field])
        data["id"] = keys
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    def dict_to_pt(data_dict, csv_path):
        pt_path = str(csv_path).replace("csv", "pt")
        with open(pt_path, "wb") as f:
            pickle.dump(data_dict, f)

    """
    Also save the csv file to hack obj:`s3prl.problem.common.run.Common`
    csvs are not used in pre-train, 
    instead we save the dict as pickle for obj:`AugmentedDynamicItemDataset` to load.
    """
    dict_to_csv(train_data, train_path)
    dict_to_csv(valid_data, valid_path)
    dict_to_csv(test_data, test_paths[0])

    dict_to_pt(train_data, train_path)
    dict_to_pt(valid_data, valid_path)
    dict_to_pt(test_data, test_paths[0])

    return train_path, valid_path, test_paths


class PretrainTera(Common):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
                train_sets=["train-clean-100", "train-clean-360", "train-other-500"],
                valid_sets=["dev-clean"],
                test_sets=["test-clean"],
            ),
            build_encoder=dict(),
            build_dataset=dict(
                target_level=-25,
                **_mask_args,
                **_noise_args,
                **_audio_config,
            ),
            build_batch_sampler=dict(
                train=dict(
                    max_length=16000 * 20,
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=2,
                ),
                test=dict(
                    batch_size=2,
                ),
            ),
            build_upstream=dict(
                config=_transformer_config,
                input_dim=_input_size,
                output_attentions=False,
                keep_multihead_output=False,
                with_input_module=True,
            ),
            build_featurizer=dict(),
            build_downstream=_transformer_config,
            build_model=dict(
                loss="torch.nn.L1Loss",
                loss_config=dict(),
            ),
            build_task=dict(),
            build_optimizer=dict(
                name="AdamW",
                conf=dict(
                    lr=2.0e-4,
                ),
            ),
            build_scheduler=dict(),
            save_model=dict(),
            save_task=dict(),
            train=dict(
                total_steps=1000000,
                log_step=25000,
                eval_step=50000,
                save_step=50000,
                gradient_clipping=5.0,
                gradient_accumulate=4,
                valid_metric="loss",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
        )

    def prepare_data(
        self,
        prepare_data: dict,
        target_dir: str,
        cache_dir: str,
        get_path_only: bool = False,
    ):
        """
        Prepare the task-specific data metadata (path, labels...).
        By default call :obj:`libri_for_pretrain` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`, support arguments in :obj:`prepare_librispeech`
            target_dir (str): Parse your corpus and save the csv file into this directory
            cache_dir (str): If the parsing or preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            get_path_only (str): Directly return the filepaths no matter they exist or not.

        Returns:
            tuple

            1. train_path (str)
            2. valid_path (str)
            3. test_paths (List[str])

            Each path (str) should be a csv file containing the following columns:

            ====================  ====================
            column                description
            ====================  ====================
            id                    (str) - the unique id for this data point
            wav_path              (str) - the absolute path of the waveform file
            transcription         (str) - a text string
            ====================  ====================
        """
        return libri_for_pretrain(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )

    def build_encoder(
        self,
        build_encoder: dict,
        target_dir: str,
        cache_dir: str,
        train_csv_path: str,
        valid_csv_path: str,
        test_csv_paths: list,
        get_path_only: bool = False,
    ):
        """
        Build the encoder (for the labels) given the data metadata, and return the saved encoder path.
        By default generate and save a :obj:`s3prl.dataio.encoder.CategoryEncoders` from all the columns
        prefixing :code:`label` from all the csv files.

        Args:
            build_encoder (dict): same in :obj:`default_config`, no argument supported for now
            target_dir (str): Save your encoder into this directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            train_csv_path (str): the train path from :obj:`prepare_data`
            valid_csv_path (str): the valid path from :obj:`prepare_data`
            test_csv_paths (List[str]): the test paths from :obj:`prepare_data`
            get_path_only (bool): Directly return the filepaths no matter they exist or not.

        Returns:
            str

            tokenizer_path: The tokenizer should be saved in the pickle format
        """
        encoder_path = Path(target_dir) / "encoder.pkl"
        if get_path_only:
            return encoder_path

        """
        Here we save a dummy encoder to hack obj:`s3prl.problem.common.run.Common`
        label encoders are not used in pre-train. 
        """
        dummy_encoder = CategoryEncoders([[""]])
        with open(encoder_path, "wb") as f:
            pickle.dump(dummy_encoder, f)

        return dummy_encoder

    def build_dataset(
        self,
        build_dataset: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        encoder_path: str,
        frame_shift: int,
    ):
        """
        Build the dataset for train/valid/test.

        Args:
            build_dataset (dict): same in :obj:`default_config`, no argument supported for now
            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): The metadata csv file for the specific :code:`mode`
            encoder_path (str): The pickled encoder path for encoding the labels

        Returns:
            torch Dataset

            For all train/valid/test mode, the dataset should return each item as a dictionary
            containing the following keys:

            ====================  ====================
            key                   description
            ====================  ====================
            x                     (torch.FloatTensor) - the masked features in (seq_len, feature_dim)
            x_len                 (int) - the waveform length :code:`seq_len`
            label                 (torch.FloatTensor) - the target features in (seq_len, feature_dim)
            label_mask            (torch.BoolTensor) - the target features' binary mask in (seq_len, feature_dim)
            position_encoding     (torch.FloatTensor) - the position encoding in (seq_len, hidden_size)
            attention_mask        (torch.FloatTensor) - the transformer attention mask in (seq_len,)
            unique_name           (str) - the unique id for this datapoint
            ====================  ====================
        """
        pt_path = str(data_csv).replace("csv", "pt")
        with Path(pt_path).open("rb") as f:
            data = pickle.load(f)
        dataset = AugmentedDynamicItemDataset(data)
        dataset = PretrainTeraPipe(**build_dataset)(dataset)
        return dataset

    def build_batch_sampler(
        self,
        build_batch_sampler: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        dataset: Dataset,
    ):
        """
        Return the batch sampler for torch DataLoader.
        By default call :obj:`superb_sid_batch_sampler` with :code:`**build_batch_sampler`.

        Args:
            build_batch_sampler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                train                 (dict) - arguments for :obj:`MaxTimestampBatchSampler`
                valid                 (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                test                  (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                ====================  ====================

            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): the :code:`mode` specific csv from :obj:`prepare_data`
            dataset: the dataset from :obj:`build_dataset`

        Returns:
            batch sampler for torch DataLoader
        """

        def _build_batch_sampler(
            train: dict = None, valid: dict = None, test: dict = None
        ):
            if mode == "train":
                wav_lens = get_info(
                    dataset, ["x_len"], Path(target_dir) / "train_stats"
                )
                return MaxTimestampBatchSampler(wav_lens, **train)
            elif mode == "valid":
                return FixedBatchSizeBatchSampler(dataset, **valid)
            elif mode == "test":
                return FixedBatchSizeBatchSampler(dataset, **test)

        return _build_batch_sampler(**build_batch_sampler)

    def build_upstream(self, build_upstream: dict):
        """
        By default build the upstream with :obj:`s3prl.nn.transformer_mockingjay.TransformerMockingjay`

        Args:
            build_upstream (dict): same in :obj:`default_config`,
                arguments for :obj:`TransformerMockingjay`

        Returns:
            :obj:`TransformerMockingjay`

            Return an upstream model for pretrain, whose forward takes the waveform input and returns
            hidden states as features for the predictor.
        """
        upstream = TransformerMockingjay(**build_upstream)
        return upstream

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_input_stride: int,
    ):
        """
        To hack obj:`s3prl.problem.common.run.Common`,
        return the predictor for pre-training as downstream model.
        By default build the :obj:`s3prl.nn.predictor_mockingjay.PredictorMockingjay` model

        Args:
            build_downstream (dict): same in :obj:`default_config`,
                support arguments of :obj:`MeanPoolingLinear`
            downstream_input_size (int): the required input size of the model
            downstream_output_size (int): the required output size of the model
            downstream_input_stride (int): the input feature's stride (from 16 KHz)

        Returns:
            :obj:`PredictorMockingjay`
        """
        predictor = PredictorMockingjay(
            config=build_downstream,
            output_dim=downstream_output_size,
            input_dim=downstream_input_size,
        )
        return predictor

    def build_model(
        self,
        build_model: dict,
        model_output_size: int,
        build_upstream: dict,
        build_featurizer: dict,
        build_downstream: dict,
    ):
        """
        By default build model with :obj:`s3prl.nn.upstream.UpstreamDownstreamModel`

        Args:
            build_model (dict): same in :obj:`default_config`,
                arguments for :obj:`s3prl.nn.upstream.UpstreamDownstreamModel`
            model_output_size (int): the required model's output hidden size
            build_upstream (dict): same in :obj:`default_config`, refer to :obj:`build_upstream`
            build_featurizer (dict): same in :obj:`default_config`, refer to :obj:`build_featurizer`
            build_downstream (dict): same in :obj:`default_config`, refer to :obj:`build_downstream`

        Returns:
            torch.nn.Module

            Return the entire model for the task, which takes the direct items from DataLoader as the input.
            Usually, the components can be built by :obj:`build_upstream`, :obj:`build_downstream`,
            and are concated together to get the final model.
            The upstream extracts hidden states,
            and the downstream (predictor) takes the hidden states as the feature to predict the pre-training objective.

            Here we take an alternative approach different from above,
            where we use :obj:`s3prl.nn.upstream.UpstreamDownstreamModel` as a wrapper to hack obj:`s3prl.problem.common.run.Common`.
            The upstream, predictor, and loss objects are simply wrapped in the final model to pass obj:`Common`,
            then dispatched and connected in the task.
            This is because different pre-training tasks may diverse a lot in terms of connecting the final model.
            So we let the task decide how to connect them rather than implementing a lot of final models.

        """
        upstream = self.build_upstream(build_upstream)
        predictor = self.build_downstream(
            build_downstream,
            downstream_input_size=None,  # None, automatically use `hidden_size` from `_transformer_config`
            downstream_output_size=_input_size,  # `output_size`` == `_input_size`` because of the reconstruction objective
            downstream_input_stride=None,  # not used
        )
        loss = eval(build_model["loss"], **build_model["loss_config"])
        model = UpstreamPredictorModel(upstream, predictor, loss)
        return model

    def build_task(
        self,
        build_task: dict,
        model: torch.nn.Module,
        encoder,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ):
        """
        Build the task, which defines the logics for every train/valid/test forward step for the :code:`model`,
        and the logics for how to reduce all the batch results from multiple train/valid/test steps into metrics

        By default build :obj:`FeatReconstructionTask`

        Args:
            build_task (dict): same in :obj:`default_config`, no argument supported for now
            model (torch.nn.Module): the model built by :obj:`build_model`
            encoder: the encoder built by :obj:`build_encoder`
            valid_df (pd.DataFrame): metadata of the valid set
            test_df (pd.DataFrame): metadata of the test set

        Returns:
            Task
        """
        return FeatReconstructionTask(model.upstream, model.predictor, model.loss)
