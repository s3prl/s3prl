import math
import copy
from typing import Callable, Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from transformer.optimization import BertAdam, Lamb


def get_optimizer(optimized_models, total_steps, optimizer_config):
    optimizer_config = copy.deepcopy(optimizer_config)
    optimizer_name = optimizer_config.pop('name')
    optimizer = eval(f'get_{optimizer_name}')(
        optimized_models,
        total_steps=total_steps,
        **optimizer_config
    )
    return optimizer


def get_grouped_parameters(optimized_models):
    named_params = []
    for m in optimized_models:
        named_params += list(m.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    return grouped_parameters


def get_BertAdam_with_schedule(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    grouped_parameters = get_grouped_parameters(optimized_models)
    optimizer = BertAdam(grouped_parameters, lr=lr,
                         warmup=warmup_proportion,
                         t_total=total_steps)
    return optimizer


def get_AdamW_with_schedule(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    grouped_parameters = get_grouped_parameters(optimized_models)
    optimizer = Lamb(grouped_parameters,
                     lr=lr,
                     warmup=warmup_proportion,
                     t_total=total_steps,
                     adam=True,
                     correct_bias=True)
    return optimizer


def get_Lamb_with_schedule(optimized_models, lr=2e-4, total_steps=20000, warmup_proportion=0.07, **kwargs):
    grouped_parameters = get_grouped_parameters(optimized_models)
    optimizer = Lamb(grouped_parameters,
                     lr=lr,
                     warmup=warmup_proportion,
                     t_total=total_steps,
                     adam=False,
                     correct_bias=False)
    return optimizer


def get_Adam(optimized_models, lr=2e-4, **kwargs):
    params = []
    for m in optimized_models:
        params += list(m.parameters())
    return Adam(params, lr=lr, betas=(0.9, 0.999))


def get_AdamW(optimized_models, lr=2e-4, **kwargs):
    params = []
    for m in optimized_models:
        params += list(m.parameters())
    optimizer = AdamW(params, lr=lr)
    return optimizer


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in
    `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
