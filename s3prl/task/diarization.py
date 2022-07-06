import numpy as np
import torch
import torch.nn as nn

from s3prl import Logs
from s3prl.base.output import Output
from s3prl.base.workspace import Workspace
from s3prl.metric.diarization import calc_diarization_error
from s3prl.nn.pit import get_label_perm, pit_loss

from .base import Task

TOLERANT_FRAME_DIFF = 2


class DiarizationPIT(Task):
    def __init__(
        self,
        model: nn.Module,
        **kwargs,
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
        predicted, predicted_len = self.model(x, x_len).slice(2)
        return Output(
            output=predicted,
            output_len=predicted_len,
        )

    def forward(self, split: str, x, x_len, label, label_len, rec_id, workspace=None, **kwds):
        predicted, predicted_len = self.predict(x, x_len).slice(2)

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

        if split == "test" and workspace is not None:
            if len(label_len) > 1:
                assert len(set(label_len[:-1].tolist())) == 1, (
                    f"Except the final chunk, other chunks from the same recording should have the same length"
                )

            predict = predicted.detach().cpu().numpy()
            # TODO:
            # predict = [p[:l] for p, l in zip(predicted.data.cpu().numpy(), predicted_len)]
            predict = np.vstack(predict)
            predict = 1 / (1 + np.exp(-predict))

            workspace = Workspace(workspace)
            rec_unique_id = set(list(rec_id))
            assert len(rec_unique_id) == 1
            (workspace / "prediction").put(predict, list(rec_unique_id)[0], "h5")

        return Output(
            loss=loss,
            accuracy=ACC,
            der=DER,
        )

    def reduction(
        self, split: str, batch_results: list, on_epoch_end: bool = None, **kwds
    ):
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
