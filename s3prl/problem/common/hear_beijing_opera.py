from omegaconf import MISSING

from .hear_esc50 import HearESC50

BEIJING_OPERA_NUM_FOLDS = 5

__all__ = ["HearBeijingOpera"]


class HearBeijingOpera(HearESC50):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
                test_fold=MISSING,
                num_folds=BEIJING_OPERA_NUM_FOLDS,
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=1,
                ),
                test=dict(
                    batch_size=1,
                ),
            ),
            build_upstream=dict(
                name=MISSING,
            ),
            build_featurizer=dict(
                layer_selections=None,
                normalize=False,
            ),
            build_downstream=dict(
                hidden_layers=2,
                pooling_type="MeanPooling",
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                prediction_type="multiclass",
                scores=["top1_acc", "d_prime", "aucroc", "mAP"],
            ),
            build_optimizer=dict(
                name="Adam",
                conf=dict(
                    lr=1.0e-3,
                ),
            ),
            build_scheduler=dict(
                name="ExponentialLR",
                gamma=0.9,
            ),
            save_model=dict(),
            save_task=dict(),
            train=dict(
                total_steps=150000,
                log_step=100,
                eval_step=1000,
                save_step=100,
                gradient_clipping=1.0,
                gradient_accumulate=1,
                valid_metric="top1_acc",
                valid_higher_better=True,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            evaluate=dict(),
        )
