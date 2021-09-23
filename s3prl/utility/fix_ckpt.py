# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utility/fix_ckpt.py ]
#   Synopsis     [ scripts to fix older checkpoints ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
Usage:
This .py helps fix the `torch serialization ModuleNotFoundError` issue, 
which occurs when the model.py directory is changed.
Make sure you understand exactly what this script does before proceeding.
"""


###############
# IMPORTATION #
###############
import os
import sys
import torch


def check_model_equiv(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        if not torch.equal(p1[0], p2[0]):
            return False
        if not torch.equal(p1[1].data, p2[1].data):
            return False
    return True


def copyParams(module_src, module_dest):
    params1 = module_src.named_parameters()
    params2 = module_dest.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data)


def main():

    input_ckpt = sys.argv[1]

    # load model with old setting
    from transformer.nn_transformer import SPEC_TRANSFORMER
    options = {'ckpt_file'     : input_ckpt,
                'load_pretrain' : 'True',
                'no_grad'       : 'True',
                'dropout'       : 'default',
                'spec_aug'      : 'False',
                'spec_aug_prev' : 'True',
                'weighted_sum'  : 'False',
                'select_layer'  : -1,
                'permute_input' : 'False' }
    old_transformer = SPEC_TRANSFORMER(options, inp_dim=-1)

    # build model with new setting
    from s3prl.upstream.mockingjay.model import TransformerForMaskedAcousticModel
    model = TransformerForMaskedAcousticModel(old_transformer.model_config, old_transformer.inp_dim, old_transformer.inp_dim).to(torch.device('cuda'))

    # load old to new
    assert not check_model_equiv(old_transformer.model, model.Transformer)
    copyParams(old_transformer.model, model.Transformer)
    assert check_model_equiv(old_transformer.model, model.Transformer)

    assert not check_model_equiv(old_transformer.SpecHead, model.SpecHead)
    copyParams(old_transformer.SpecHead, model.SpecHead)
    assert check_model_equiv(old_transformer.SpecHead, model.SpecHead)

    global_step = old_transformer.all_states['Global_step']
    settings = old_transformer.all_states['Settings']

    # save
    all_states = {
        'SpecHead': model.SpecHead.state_dict(),
        'Transformer': model.Transformer.state_dict(),
        'Global_step': global_step,
        'Settings': settings
    }
    new_ckpt_path = input_ckpt.replace('.ckpt', '-new.ckpt')
    torch.save(all_states, new_ckpt_path)

    print('Done fixing ckpt: ', input_ckpt, 'to: ', new_ckpt_path)


if __name__ == '__main__':
    main()
