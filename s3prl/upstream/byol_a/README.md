# BYOL-A upstream model for SUPERB

- BYOL-A reqires normalization statistics pre-computed for each downstream tasks.
- Evaluate BYOL-A in two steps. First, calculate statistics, then, train and test using the computed stats.

## Using BYOL-A on SUPERB

### 1. Pre-compute statistics.

The followings explain using a downstream task `voxceleb1`.

    python run_downstream.py -m train -n byol_a_2048_calcnorm_1 -u byol_a_2048_calcnorm -d voxceleb1

This will output:

    ** Running Norm has finished updates over 10000 times, using the following stats from now on. ***
    mean=-8.907230377197266, std=4.892485618591309
    *** Please use these statistics in your model. EXIT... ***

These `-8.907230377197266` and `4.8924856` are the statistics for the `voxceleb1`. We add a function for using these values in the `hubconf.py`.

```python
def byol_a_2048_vc1(refresh=False, **kwds):
    """BYOL-A d=2048 for voxceleb1."""
    return _byol_a_2048(norm_mean=-8.9072303, norm_std=4.8924856, **kwds)
```

### 2. Evaluating on a downstream task

For the `voxceleb1`, we use the `byol_a_2048_vc1` to train and test.

    python run_downstream.py -m train -n byol_a_2048_vc1_1 -u byol_a_2048_vc1 -d voxceleb1
    python run_downstream.py -m train -n byol_a_2048_vc1_1 -u byol_a_2048_vc1 -d voxceleb1 -o "config.optimizer.lr=1e-2"

The following will test the trained model.

    python run_downstream.py -m evaluate -n byol_a_2048_vc1_1 -d voxceleb1 -e result/downstream/byol_a_2048_vc1_1/dev-best.ckpt


## Examples

### PR

    pip install editdistance

    python run_downstream.py -m train -n byol_a_2048_pr_1 -u byol_a_2048_LS -d ctc -c downstream/ctc/libriphone.yaml -o "config.optimizer.lr=1e-3"
    CUDA_VISIBLE_DEVICES=1 python run_downstream.py -m train -n byol_a_2048_pr_1 -u byol_a_2048_LS -d ctc -c downstream/ctc/libriphone.yaml -o "config.optimizer.lr=1e-3"

    CUDA_VISIBLE_DEVICES=1 python run_downstream.py -m evaluate -n byol_a_2048_pr_1 -d ctc -c downstream/ctc/libriphone.yaml -e result/downstream/byol_a_2048_pr_1/dev-best.ckpt

