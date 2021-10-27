import io
import os
import sys
from typing import Optional, Union
from pytorch_lightning.callbacks.progress import ProgressBar, tqdm, convert_inf, reset


class LitProgressBar(ProgressBar):
    """ Customized Lightning ProgressBar.
        The default Lightning ProgressBar does not include global step-wise
        progress bar, but only epoch-wise bars.
        This class adds a global progress bar on top of Lightning default ProgressBar.
    """
    def __init__(self, refresh_rate: int = 1, process_position: int = 1):
        assert process_position > 0, "Require space for global progress bar"
        super().__init__(refresh_rate, process_position)
        self.global_progress_bar = None
        self._train_global_idx = 0


    def __getstate__(self):
        # can't pickle the tqdm objects
        state = super().__getstate__()
        state["global_progress_bar"] = None
        return state


    @property
    def train_global_idx(self):
        return self._train_global_idx


    def init_global_tqdm(self):
        bar = tqdm(
            desc="Overall",
            initial=self.train_global_idx,
            position=(2 * self.process_position - 1),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar


    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.global_progress_bar = self.init_global_tqdm()
        reset(self.global_progress_bar, trainer.max_steps)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._train_global_idx += 1

        total_batches = convert_inf(trainer.max_steps)
        if self._should_update(self.train_global_idx, total_batches):
            self._update_bar(self.global_progress_bar)


    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.global_progress_bar.close()


    def print(
        self, *args, sep: str = " ", end: str = os.linesep, file: Optional[io.TextIOBase] = None, nolock: bool = False
    ):
        active_progress_bar = None

        if self.global_progress_bar is not None and not self.global_progress_bar.disable:
            active_progress_bar = self.global_progress_bar
        elif self.main_progress_bar is not None and not self.main_progress_bar.disable:
            active_progress_bar = self.main_progress_bar
        elif self.val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self.test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self.predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, end=end, file=file, nolock=nolock)

