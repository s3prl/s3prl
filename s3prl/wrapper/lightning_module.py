import logging
from collections import OrderedDict

from pytorch_lightning import LightningModule
from s3prl import Output, LogDataType
from s3prl.task import Task

logger = logging.getLogger(__name__)


class LightningModuleSimpleWrapper(LightningModule):
    def __init__(self, task: Task, optimizer) -> None:
        super().__init__()
        self.task = task
        self.optimizer = optimizer

    def _common_step(self, batch, step_fn, prefix: str):
        output: Output = step_fn(**batch)
        output.prefix = prefix
        return OrderedDict(**output)

    def _log_output(self, output: Output):
        for key, value in output.get(f"logs", {}).items():
            if value.data_type == LogDataType.SCALAR:
                self.log(f"{output.prefix}_{key}", value.data)
                logger.info(f"{output.prefix}_{key}: {value.data}")

    def _common_reduction(self, step_outputs, reduction_fn):
        assert len(step_outputs) > 0
        multiple_dataloader = isinstance(step_outputs[0], list)
        if not multiple_dataloader:
            step_outputs = [step_outputs]
        multiple_dataloader_outputs = step_outputs

        for single_dataloader_outputs in multiple_dataloader_outputs:
            step_outputs = [Output(**output) for output in single_dataloader_outputs]
            prefix = step_outputs[0].prefix
            step_outputs = [output.deselect("prefix") for output in step_outputs]
            result: Output = reduction_fn(step_outputs)
            result.prefix = prefix
            self._log_output(result)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, self.task.train_step, "train")

    def training_epoch_end(self, step_outputs):
        self._common_reduction(step_outputs, self.task.train_reduction)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._common_step(
            batch, self.task.valid_step, f"valid_{dataloader_idx or 0}"
        )

    def validation_epoch_end(self, step_outputs):
        self._common_reduction(step_outputs, self.task.valid_reduction)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._common_step(
            batch, self.task.test_step, f"test_{dataloader_idx or 0}"
        )

    def test_epoch_end(self, step_outputs):
        self._common_reduction(step_outputs, self.task.test_reduction)

    def configure_optimizers(self):
        return self.optimizer
