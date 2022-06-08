"""
#   Source       Refactored from https://github.com/hitachi-speech/EEND
#   Author       Jiatong Shi, Leo Yang
#   Copyright    Copyright(c), Johns Hopkins University, National Taiwan University
"""

import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn

from s3prl import Logs
from s3prl.base.output import Output
from s3prl.base.workspace import Workspace
from s3prl.metric.pit import pit_loss, get_label_perm
from s3prl.metric.diar import calc_diarization_error
from .base import Task

TOLERANT_FRAME_DIFF = 2


class DiarizationPIT(Task):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self,
        model: nn.Module,
        workspace: Workspace = None,
        save_prediction_to: str = "prediction",
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.objective = pit_loss
        self.prediction_dir = None
        if workspace is not None:
            self.prediction_dir = Workspace(workspace) / save_prediction_to

    def _tile_representations(self, reps, factor):
        """
        Tile up the representations by `factor`.
        Input - sequence of representations, shape: (batch_size, seq_len, feature_dim)
        Output - sequence of tiled representations, shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(reps.shape) == 3
        ), "Input argument `reps` has invalid shape: {}".format(reps.shape)
        tiled_reps = reps.repeat(1, 1, factor)
        tiled_reps = tiled_reps.reshape(
            reps.size(0), reps.size(1) * factor, reps.size(2)
        )
        return tiled_reps

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(1)

        factor = int(round(label_len / input_len))
        if factor > 1:
            inputs = self._tile_representations(inputs, factor)
            input_len = inputs.size(1)

        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            pad_vec = inputs[:, -1, :].unsqueeze(1)  # (batch_size, 1, feature_dim)
            inputs = torch.cat(
                (inputs, pad_vec.repeat(1, label_len - input_len, 1)), dim=1
            )  # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
        return inputs, labels

    def predict(self, x, x_len):
        predicted = self.model(x, x_len)
        return predicted

    def forward(self, split: str, x, x_len, label, rec_id, **kwargs):
        predicted, _ = self.predict(x, x_len)

        assert (
            abs(predicted.size(1) - label.size(1)) <= TOLERANT_FRAME_DIFF
        ), f"predicted: {predicted.shape}, label: {label.shape}, TOLERANT_FRAME_DIFF: {TOLERANT_FRAME_DIFF}"

        predicted, label = self._match_length(predicted, label)
        loss, perm_idx, perm_list = self.objective(predicted, label.float(), x_len)
        label_perm = get_label_perm(label, perm_idx, perm_list)

        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = calc_diarization_error(predicted, label_perm, x_len)

        if speech_scored > 0 and speaker_scored > 0 and num_frames > 0:
            SAD_MR, SAD_FR, MI, FA, CF, ACC, DER = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            SAD_MR, SAD_FR, MI, FA, CF, ACC, DER = 0, 0, 0, 0, 0, 0, 0

        if split == "test" and self.prediction_dir is not None:
            predict = predicted.data.cpu().numpy()
            predict = np.vstack(list(predict))
            predict = 1 / (1 + np.exp(-predict))
            rec_unique_id = set(list(rec_id))
            assert len(rec_unique_id) == 1
            self.prediction_dir.put(predict, list(rec_unique_id)[0], "h5")

        return Output(
            loss=loss,
            accuracy=ACC,
            der=DER,
        )

    def reduction(self, split: str, batch_results: list, on_epoch_end: bool = None):
        accs, ders = [], []
        for batch_result in batch_results:
            accs.append(batch_result.accuracy)
            ders.append(batch_result.der)

        logs = Logs()
        logs.add_scalar("accuracy", torch.FloatTensor(accs).mean().item())
        logs.add_scalar("der", torch.FloatTensor(ders).mean().item())

        return Output(
            logs=logs,
        )
