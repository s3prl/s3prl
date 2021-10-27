import os
import glob
from tqdm import tqdm
from collections import defaultdict

import torch.distributed as dist
from pytorch_lightning.callbacks import Callback


class Saver(Callback):
    def __init__(self,):
        super().__init__()


    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.train_records = defaultdict(list)
        self.train_batch_ids = []


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        global_step = pl_module.global_step + 1
        logger = pl_module.logger.experiment
        config = pl_module.config
        args = pl_module.args

        if trainer.is_global_zero:
            self.train_batch_ids.append(batch_idx)
            for k, v in outputs['records'].items():
                self.train_records[k] += v

            # Logging
            if global_step % config['runner']['log_step'] == 0:
                pl_module.downstream.model.log_records(
                    'train',
                    records = self.train_records,
                    logger = logger,
                    global_step = global_step,
                    batch_ids = self.train_batch_ids,
                    total_batch_num = len(pl_module.train_dataloader),
                )
                self.train_batch_ids = []
                self.train_records = defaultdict(list)

            # Save checkpoint
            if global_step % config['runner']['save_step'] == 0:
                def check_ckpt_num(directory):
                    max_keep = config['runner']['max_keep']
                    ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                    if len(ckpt_pths) >= max_keep:
                        ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                        for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                            os.remove(ckpt_pth)
                check_ckpt_num(args.expdir)
                trainer.save_checkpoint(f'states-{global_step}.ckpt')


    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.validation_records = defaultdict(lambda: defaultdict(list))
        self.validation_batch_ids = defaultdict(list)


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        split = pl_module.config['runner']['eval_dataloaders'][dataloader_idx]
        world_size = dist.get_world_size()

        # Synchronize outputs across GPUs
        dist.barrier()
        outputs_gather = [None for _ in range(world_size)]
        dist.all_gather_object(outputs_gather, outputs)

        if trainer.is_global_zero:
            for i, output in enumerate(outputs_gather):
                # Treat sub-batch on each GPU an individual batch
                batch_id = batch_idx * world_size + i
                self.validation_batch_ids[split].append(batch_id)
                for k, v in output['records'].items():
                    self.validation_records[split][k] += v


    def validation_epoch_end(self, trainer, pl_module):
        global_step = pl_module.global_step + 1
        logger = pl_module.logger.experiment
        config = pl_module.config
        args = pl_module.args

        if trainer.is_global_zero:
            save_names = []
            for split in config['runner']['eval_dataloaders']:
                save_names += pl_module.downstream.model.log_records(
                    split,
                    records = self.validation_records[split],
                    logger = logger,
                    global_step = global_step,
                    batch_ids = self.validation_batch_ids[split],
                    total_batch_num = len(self.validation_batch_ids[split]),
                )

            if len(save_names) > 0:
                save_paths = [os.path.join(args.expdir, name) for name in save_names]
                tqdm.write(f'[Runner] - Save the checkpoint to:')
                for i, path in enumerate(save_paths):
                    tqdm.write(f'{i + 1}. {path}')
                    trainer.save_checkpoint(path)

        self.validation_records = defaultdict(lambda: defaultdict(list))
        self.validation_batch_ids = defaultdict(list)

