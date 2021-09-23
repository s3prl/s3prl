# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utility/run_sig_test.py ]
#   Synopsis     [ Run the significance test on TWO RELATED samples of scores, a and b, from two experiment checkpoints ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
Note:
In order to calculate the significance tests,
the downstream expert's forward() method should log a metric score for each testing sample.
The `records['sample_wise_metric']` should be a list containing the testing result of each sample.

For example:
```
python
# for frame-wise classification
for sample in samples:
    records['sample_wise_metric'] += [torch.FloatTensor(sample).mean().item()]
# for utterance-wise classification
records['sample_wise_metric'] += (predicted_classid == labels).view(-1).cpu().tolist()
```
"""


###############
# IMPORTATION #
###############
import os
import glob
import random
import argparse
from tqdm import tqdm
from argparse import Namespace
#-------------#
import torch
import torchaudio
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
#-------------#
from s3prl.downstream.runner import Runner
from s3prl.utility.helper import hack_isinstance, override, defaultdict


def get_ttest_args():
    parser = argparse.ArgumentParser()
    # ttest, for continuous result: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html#scipy.stats.ttest_rel
    # fisher, for categorical results: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    # mcnemar, for categorical results: https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html
    parser.add_argument('-m', '--mode', choices=['ttest', 'fisher', 'mcnemar'], default='ttest')
    
    parser.add_argument('-em', '--evaluate_metric', default='acc')
    parser.add_argument('-t', '--evaluate_split', default='test')
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # compare two ckpts with the Paired Sample T-test using SciPy
    # All the args and config ill be determined by the ckpts
    # if a directory is specified, the latest ckpt will be used by default
    parser.add_argument('-e1', '--past_exp1', metavar='{CKPT_PATH,CKPT_DIR}', help='Load from a checkpoint')
    parser.add_argument('-e2', '--past_exp2', metavar='{CKPT_PATH,CKPT_DIR}', help='Load from another checkpoint')
    parser.add_argument('-u1', '--upstream1', default='default', type=str, help='used to override the upstream string for checkpoint e1')
    parser.add_argument('-u2', '--upstream2', default='default', type=str, help='used to override the upstream string for checkpoint e2')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--verbose', action='store_true', help='Print model infomation')
    parser.add_argument('--ckpt_name', default='best-states-dev', \
                        help='The string used for searching the checkpoint, \
                        example choices: `states-*`, `best-states-dev`, `best-states-test`.')
    args = parser.parse_args()

    args1, config1 = get_past_exp(args, args.past_exp1, args.ckpt_name)
    args2, config2 = get_past_exp(args, args.past_exp2, args.ckpt_name)
    if args.upstream1 != 'default': args1.upstream = args.upstream1
    if args.upstream2 != 'default': args2.upstream = args.upstream2

    return args.mode, args1, config1, args2, config2


def get_past_exp(args, past_exp, name):
    # determine checkpoint path
    if os.path.isdir(past_exp):
        ckpt_pths = glob.glob(os.path.join(past_exp, f'{name}.ckpt'))
        assert len(ckpt_pths) > 0
        if len(ckpt_pths) == 1:
            ckpt_pth = ckpt_pths[0]
        else:
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
    else:
        ckpt_pth = past_exp

    print(f'[Runner] - Loading from {ckpt_pth}')

    # load checkpoint
    ckpt = torch.load(ckpt_pth, map_location='cpu')

    def update_args(old, new, preserve_list=None):
        out_dict = vars(old)
        new_dict = vars(new)
        for key in list(new_dict.keys()):
            if key in preserve_list:
                new_dict.pop(key)
        out_dict.update(new_dict)
        return Namespace(**out_dict)

    # overwrite args
    cannot_overwrite_args = [
        'mode', 'evaluate_split', 'override',
        'backend', 'local_rank', 'past_exp',
    ]
    args = update_args(args, ckpt['Args'], preserve_list=cannot_overwrite_args)

    args.init_ckpt = ckpt_pth
    args.mode = 'evaluate'
    config = ckpt['Config']

    if args.override:
        override(args.override, args, config)
    return args, config


class Tester(Runner):
    """
    Used to handle the evaluation loop and return the testing records for Paired Sample T-test.
    """
    def __init__(self, args, config):
        super(Tester, self).__init__(args, config)
    
    def evaluate(self):
        """evaluate function will always be called on a single process even during distributed training"""

        split = self.args.evaluate_split

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.cuda.device(self.args.device):
            torch.cuda.empty_cache()

        # set all models to eval
        self.downstream.eval()
        self.upstream.eval()

        # prepare data
        dataloader = self.downstream.get_dataloader(split)

        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split)):

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream(wavs)
                self.downstream(
                    split,
                    features, *others,
                    records = records,
                )
        return records


def process_records(records, metric):
    assert 'sample_wise_metric' in records, 'Utterance-wise / sample-wise metric is necessary for proceeding the Paired Sample T-test.'
    average = torch.FloatTensor(records[metric]).mean().item()
    return average, records['sample_wise_metric']


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('sox_io')
    hack_isinstance()

    # get config and arguments
    mode, args1, config1, args2, config2 = get_ttest_args()

    # Fix seed and make backends deterministic
    random.seed(args1.seed)
    np.random.seed(args1.seed)
    torch.manual_seed(args1.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args1.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tester1 = Tester(args1, config1)
    records1 = eval(f'tester1.{args1.mode}')()
    average1, sample_metric1 = process_records(records1, args1.evaluate_metric)

    tester2 = Tester(args2, config2)
    records2 = eval(f'tester2.{args2.mode}')()
    average2, sample_metric2 = process_records(records2, args2.evaluate_metric)

    if mode == 'ttest':
        statistic, p_value = stats.ttest_rel(sample_metric1, sample_metric2)
    elif mode == 'fisher':
        correct1 = sample_metric1.count(True)
        correct2 = sample_metric2.count(True)
        contingency_table = [[correct1, correct2], 
                             [len(sample_metric1)-correct1, len(sample_metric2)-correct2]]
        statistic, p_value = stats.fisher_exact(contingency_table)
    elif mode == 'mcnemar':
        correct1 = sample_metric1.count(True)
        correct2 = sample_metric2.count(True)
        contingency_table = [[correct1, correct2], 
                             [len(sample_metric1)-correct1, len(sample_metric2)-correct2]]
        b = mcnemar(contingency_table, exact=True)
        statistic, p_value = b.statistic, b.pvalue
    else:
        raise NotImplementedError
    
    print(f'[Runner] - The testing scores of the two ckpts are {average1} and {average2}, respectively.')
    print(f'[Runner] - The statistic of the significant test of the two ckpts is {statistic}')
    print(f'[Runner] - The P value of significant test of the two ckpts is {p_value}')


if __name__ == '__main__':
    main()
