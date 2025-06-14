# BYOL-A upstream model for SUPERB

This is the upstream model of BYOL-A.

**Note** that you will need to compute the statistics (i.e., mean and standard deviation) of the downstream task before training and testing on it, because BYOL-A requires pre-computed normalization statistics for each downstream task.

## Usage

Install nnAudio if not.

    pip install nnAudio

### 0. Downloading weights

Download the weight from the links below.

- 2048-d feature (recommended): https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth
- 1024-d feature: https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d1024.pth
- 512-d feature: https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d512.pth

### 1. Configuring the number of feature dimensions

Edit your `config.yaml` to match your weight.

    # Dimensions of feature representations.
    feature_d: 2048

### 2. Pre-computing statistics

This is required for each downstream task. Once you have done this, you do not need to calculate again; keep the calculated statistics for the task.
The followings is an example for `voxceleb1`.

    python run_downstream.py -m train -n byola_calcnorm_vc1 -u byol_a_calcnorm -d voxceleb1 -k ./AudioNTT2020-BYOLA-64x96d2048.pth

This will output:

    ** Running Norm has finished updates over 10000 times, using the following stats from now on. ***
    mean,std=-8.907230377197266,4.892485618591309
    *** Please use these statistics in your model. EXIT... ***

These `-8.907230377197266` and `4.892485618591309` are the statistics for the `voxceleb1`.

### 3. Training and testing

The followings is an example for `voxceleb1`.

    python run_downstream.py -m train -n my_byol_a_vc1_1 -u byol_a -d voxceleb1 -o "config.optimizer.lr=1e-3" -k ./AudioNTT2020-BYOLA-64x96d2048.pth,-8.907230377197266,4.892485618591309
    python run_downstream.py -m evaluate -e result/downstream/my_byol_a_vc1_1/dev-best.ckpt

As you can see, `-o "config.optimizer.lr=1e-3"` sets the learning rate.

## Examples

### PR

    pip install editdistance

    python run_downstream.py -m train -n byol_a_2048_pr_1 -u byol_a -d ctc -c downstream/ctc/libriphone.yaml -o "config.optimizer.lr=1e-2" -k ./AudioNTT2020-BYOLA-64x96d2048.pth,-8.5402,4.5456
    python run_downstream.py -m evaluate -e result/downstream/byol_a_2048_pr_1/dev-best.ckpt


