"""
Diarization Permutation Invarant Task

Authors
  * Jiatong Shi 2022
  * Leo 2022
"""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from s3prl.metric.diarization import calc_diarization_error
from s3prl.nn.pit import get_label_perm, pit_loss

from .base import Task

TOLERANT_FRAME_DIFF = 2

__all__ = ["DiarizationPIT"]


class DiarizationPIT(Task):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.objective = pit_loss

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
        predicted, predicted_len = self.model(x, x_len)
        return predicted, predicted_len

    def forward(
        self,
        _mode: str,
        x,
        x_len,
        label,
        label_len,
        record_id: str,
        chunk_id: int,
        _dump_dir: str = None,
    ):
        predicted, predicted_len = self.predict(x, x_len)

        for pl, ll in zip(predicted_len, label_len):
            assert (
                abs(pl - ll) <= TOLERANT_FRAME_DIFF
            ), f"predicted: {pl}, label: {ll}, TOLERANT_FRAME_DIFF: {TOLERANT_FRAME_DIFF}"

        predicted, label = self._match_length(predicted, label)
        loss, perm_idx, perm_list = self.objective(predicted, label.float(), label_len)
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
        ) = calc_diarization_error(predicted, label_perm, label_len)

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

        if _mode == "test" and _dump_dir is not None:
            assert (
                len(set(list(record_id))) == 1
            ), "During testing, all utterances in a batch should come from the same recording"

            if len(label_len) > 1:
                assert (
                    len(set(label_len[:-1].tolist())) == 1
                ), f"Except the final chunk, other chunks from the same recording should have the same length"

            predicted_sorted = []
            for idx in chunk_id.long().topk(len(chunk_id), largest=False).indices:
                predicted_sorted.append(predicted[idx])

            predict = torch.vstack(predicted_sorted)
            predict = predict.detach().cpu()
            predict = 1 / (1 + (-predict).exp())

            prediction_dir = Path(_dump_dir) / f"prediction"
            prediction_dir.mkdir(exist_ok=True, parents=True)
            torch.save(predict, prediction_dir / f"{record_id[0]}.pt")

        cacheable = dict(
            loss=loss.detach().cpu(),
            accuracy=ACC,
            der=DER,
        )

        return loss, cacheable

    def reduction(self, _mode: str, cached_results: List[dict], _dump_dir: str = None):
        results = self.parse_cached_results(cached_results)
        logs = dict(
            accuracy=torch.FloatTensor(results["accuracy"]).mean().item(),
            der=torch.FloatTensor(results["der"]).mean().item(),
        )
        return logs
