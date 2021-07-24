import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self, set_to_none=True):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives, grad_list=None, shape_list=None, has_grad_list=None,
                    is_pcgrad=True, per_layer=True):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives, 
                                                   grad_list, shape_list, has_grad_list, per_layer=per_layer)
        pc_grad, total_counts, conflict_counts, condition_a_counts = \
                self._project_conflicting(grads, has_grads, is_pcgrad, per_layer=per_layer)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0], per_layer=per_layer)
        self._set_grad(pc_grad)
        return total_counts, conflict_counts, condition_a_counts

    def _project_conflicting(self, grads, has_grads, is_pcgrad, shapes=None, per_layer=True):
        if per_layer:
            shared = [torch.stack([h_g[l] for h_g in has_grads]).prod(0).bool() for l in range(len(has_grads[0]))]
        else:
            shared = torch.stack(has_grads).prod(0).bool()

        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        total_counts = 0
        conflict_counts = 0
        condition_a_counts = 0
        if per_layer:
            for i, g_i in enumerate(pc_grad):
                random.shuffle(grads)
                for g_j in grads:
                    for l, (g_il, g_jl) in enumerate(zip(g_i, g_j)):
                        g_il_g_jl = torch.dot(g_il, g_jl)
                        total_counts += 1
                        if g_il_g_jl < 0:
                            conflict_counts += 1
                            g_proj = (g_il_g_jl) * g_jl / (g_jl.norm()**2)
                            if torch.isnan(g_proj).sum() > 0:
                                #print('g_proj is nan and skip...')
                                continue
                            if g_il_g_jl <= -2*(g_il.norm()**2)*(g_jl.norm()**2)/((g_il.norm()**2)+(g_jl.norm()**2)):
                                condition_a_counts += 1
                            if is_pcgrad:
                                g_il -= g_proj
                                pc_grad[i][l] = g_il
                total_counts -= len(g_i)
        else:
            for i, g_i in enumerate(pc_grad):
                random.shuffle(grads)
                for g_j in grads:
                    g_i_g_j = torch.dot(g_i, g_j)
                    total_counts += 1
                    if g_i_g_j < 0:
                        conflict_counts += 1
                        g_proj = (g_i_g_j) * g_j / (g_j.norm()**2)
                        if torch.isnan(g_proj).sum() > 0:
                            #print('g_proj is nan and skip...')
                            continue
                        if g_i_g_j <= -2*(g_i.norm()**2)*(g_j.norm()**2)/((g_i.norm()**2)+(g_j.norm()**2)):
                            condition_a_counts += 1
                        if is_pcgrad:
                            g_i -= g_proj
                            pc_grad[i] = g_i
                total_counts -= 1

        if per_layer:
            merged_grad = [torch.zeros_like(g).to(g.device) for g in grads[0]]
            for l, shared_l in enumerate(shared):
                merged_grad[l][shared_l] = torch.stack([g[l][shared_l]
                                                   for g in pc_grad]).mean(dim=0)
                merged_grad[l][~shared_l] = torch.stack([g[l][~shared_l]
                                                    for g in pc_grad]).sum(dim=0)
        else:
            merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
            merged_grad[~shared] = torch.stack([g[~shared]
                                                for g in pc_grad]).sum(dim=0)

        return merged_grad, total_counts, conflict_counts, condition_a_counts

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, grads=None, shapes=None, has_grads=None, per_layer=True):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        if grads is None:
            random.shuffle(objectives)
            grads, shapes, has_grads = [], [], []
            for obj in objectives:
                self._optim.zero_grad(set_to_none=True)
                obj.backward(retain_graph=True)
                grad, shape, has_grad = self._retrieve_grad()
                grads.append(self._flatten_grad(grad, shape))
                has_grads.append(self._flatten_grad(has_grad, shape))
                shapes.append(shape)
        else:
            lists = list(zip(grads, shapes, has_grads))
            random.shuffle(lists)
            grads, shapes, has_grads = zip(*lists)
            grads = list(grads)
            shapes = list(shapes)
            has_grads = list(has_grads)
            for i, (grad, shape, has_grad) in enumerate(zip(grads, shapes, has_grads)):
                grads[i] = self._flatten_grad(grad, shape, per_layer=per_layer)
                has_grads[i] = self._flatten_grad(has_grad, shape, per_layer=per_layer)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes, per_layer=True):
        unflatten_grad, idx = [], 0
        for i, shape in enumerate(shapes):
            if per_layer:
                unflatten_grad.append(grads[i].view(shape).clone())
            else:
                length = np.prod(shape)
                unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
                idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes, per_layer=True):
        if per_layer:
            flatten_grad = [g.flatten() for g in grads]
        else:
            flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    total_norm = 0.
    for p in net.parameters():
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
