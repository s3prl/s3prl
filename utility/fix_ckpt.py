# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utility/fix_ckpt.py ]
#   Synopsis     [ scripts to fix older checkpoints ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
Usage:
This .py helps fix the old checkpoint name issue.
1) First create a dummy directory called "utils/"
2) Copy utility/timer.py to utils/timer.py
3) Download the old "mockingjay/" directory from github history: https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/tree/f87fe59ad08eb399006fcba4f2ecc0bb72f9781f
4) Copy transformer/mam.py to utils/mam.py
5) Run this script to fix the old checkpoint files.
"""


###############
# IMPORTATION #
###############
import os
import sys
import torch


input_ckpt = sys.argv[1]
all_states = torch.load(input_ckpt, map_location='cpu')


# handling config attribute of older checkpoints
config = all_states['Settings']['Config']
for cluster in config:
    if 'timer' in config:
        del config['timer']
        print('[Fixer] - Deleted `timer` attribute in config.')
        break
# handling config attribute of older checkpoints
attribute = 'transformer' if 'transformer' in config else 'mockingjay'
if 'prune_headids' not in config[attribute]:
    config[attribute]['prune_headids'] = None
    print('[Fixer] - Added `prune_headids` attribute in config.')
# handling config attribute of older checkpoints
if 'input_dim' not in config[attribute]:
    dim = int(input('Please enter input dim: '))
    config[attribute]['input_dim'] = dim
    print('[Fixer] - Added `input_dim` attribute in config.')
# handling config attribute of older checkpoints
if 'test_reconstruct' in config[attribute]:
    del config[attribute]['test_reconstruct']
    print('[Fixer] - Deleted `test_reconstruct` attribute in config.')


# handling paras of older checkpoints
paras = all_states['Settings']['Paras']
if not hasattr(paras, 'multi_gpu'):
    setattr(paras, 'multi_gpu', False)


# load model with mockingjay Trainer
try:
    from mockingjay.solver import Trainer as m_Trainer
    m_trainer = m_Trainer(config, paras)
except:
    print('[Fixer] - Aborting, failed to build mockingjay trainer...')
    exit()

try:    
    m_trainer.load = True
    m_trainer.load_model_list = ['SpecHead', 'Mockingjay', 'Optimizer', 'Global_step']
    m_trainer.set_model(inference=False, from_path=input_ckpt)
except:
    print('[Fixer] - Aborting, failed to load with mockingjay trainer...')
    exit()


# handle deprecated naming for older config files
if 'mockingjay' in config:
    config['transformer'] = config['mockingjay']
    del config['mockingjay']
    print('[Fixer] - Renamed `mockingjay` attribute to `transformer` in config.')


# load model with transformer Trainer
try:
    import copy
    from transformer.solver import Trainer as t_Trainer
    t_paras = copy.deepcopy(paras)
    t_paras.verbose = False
    t_trainer = t_Trainer(config, t_paras)
except:
    print('[Fixer] - Aborting, failed to build transformer trainer...')
    exit()

try: 
    t_trainer.load = False
    t_trainer.set_model(inference=False)
except:
    print('[Fixer] - Aborting, failed to set model with transformer trainer...')
    exit()

while True:
    proceed = str(input('Do you want to proceed? [y/n]: '))
    if proceed == 'n': exit()
    elif proceed == 'y': break
    else: pass

def check_model_equiv(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        if not torch.equal(p1[0], p2[0]):
            return False
        if not torch.equal(p1[1].data, p2[1].data):
            return False
    return True

# handling path and name of older checkpoints
check1 = check_model_equiv(t_trainer.model, m_trainer.model)
check2 = check_model_equiv(t_trainer.transformer, m_trainer.mockingjay)
print('[Fixer] - Model weight equivalent before weight transfer: ', check1)
print('[Fixer] - Transformer weight equivalent before weight transfer: ', check2)
if check1 or check2:
    print('[Fixer] - Aborting, model weights are alreday identical...')
    exit()

t_trainer.model = m_trainer.model
t_trainer.transformer = m_trainer.mockingjay
t_trainer.optimizer = m_trainer.optimizer
t_trainer.global_step = m_trainer.global_step
t_trainer.paras.verbose = True
print('[Fixer] - Model weight equivalent before weight transfer: ', check_model_equiv(t_trainer.model, m_trainer.model))
print('[Fixer] - Transformer weight equivalent after weight transfer: ', check_model_equiv(t_trainer.transformer, m_trainer.mockingjay))

os.remove(input_ckpt)
if '/mockingjay-' in input_ckpt: input_ckpt = input_ckpt.replace('/mockingjay-', '/model-')
t_trainer.save_model(to_path=input_ckpt)
print('Done fixing ckpt: ', input_ckpt)
exit()
