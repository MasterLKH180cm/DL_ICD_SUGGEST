import argparse
import glob
import itertools as it
import math
import os
import pdb
import pickle as pkl
import sys
import traceback
from collections import OrderedDict
from multiprocessing.reduction import ForkingPickler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import (coverage_error, f1_score, hamming_loss,
                             label_ranking_average_precision_score,
                             label_ranking_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.multiprocessing import reductions
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import Dataset, dataloader

default_collate_func = dataloader.default_collate

version_higher = torch.__version__ >= "1.5.0"


class AdaBelief(Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-16,
                 weight_decay=0,
                 amsgrad=False,
                 weight_decouple=True,
                 fixed_decay=False,
                 rectify=True,
                 degenerated_to_sgd=True):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params,
                      (list, tuple)) and len(params) > 0 and isinstance(
                          params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0]
                                         or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        buffer=[[None, None, None] for _ in range(10)])
        super(AdaBelief, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMSGrad enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format
                ) if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format
                ) if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    ) if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    ) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    ) if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad.
                        # values
                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        ) if version_higher else torch.zeros_like(p.data)

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual,
                                                 grad_residual,
                                                 value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_var,
                              exp_avg_var,
                              out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2**state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * \
                         state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                                (N_sma - 2) / N_sma * N_sma_max /
                                (N_sma_max - 2)) / (1 - beta1**state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1**state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg,
                                        denom,
                                        value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

        return loss


class Ranger(Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 alpha=0.5,
                 k=6,
                 N_sma_threshhold=5,
                 betas=(.95, 0.999),
                 eps=1e-5,
                 weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to
        # make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr,
                        alpha=alpha,
                        k=k,
                        step_counter=0,
                        betas=betas,
                        N_sma_threshhold=N_sma_threshhold,
                        eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        # removed as we now use step from RAdam...no need for duplicate step counting
        # for group in self.param_groups:
        #    group["step_counter"] = 0
        # print("group step counter init")

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # self.first_run_check=0

        # lookahead weights
        # 9/2/19 - lookahead param tensors have been moved to state storage.
        # This should resolve issues with load/save where weights were left in
        # GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        # don't use grad for lookahead weights
        # for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(
                        state
                ) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved
                    # model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1

                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2**state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                     state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max /
                            (N_sma_max - 2)) / (1 - beta1**state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1**state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                     p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg,
                                         denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get f1ess to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(self.alpha, p.data - slow_p)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss


class Model(nn.Module):

    def __init__(self, atc_start, pro_start, icd_start, atc_dim, pro_dim,
                 icd_dim):
        super(Model, self).__init__()
        """ declare layers used in this network"""
        self.atc_start = atc_start
        self.pro_start = pro_start
        self.icd_start = icd_start
        self.dim_agg = 128

        self.pro_embeds = nn.Embedding(pro_dim,
                                       self.dim_agg,
                                       max_norm=True,
                                       scale_grad_by_freq=True)
        self.atc_embeds = nn.Embedding(atc_dim,
                                       self.dim_agg,
                                       max_norm=True,
                                       scale_grad_by_freq=True)
        self.icd_embeds = nn.Embedding(icd_dim,
                                       self.dim_agg,
                                       max_norm=True,
                                       scale_grad_by_freq=True)
        self.sex_embeds = nn.Embedding(2,
                                       2,
                                       max_norm=True,
                                       scale_grad_by_freq=True)
        self.age_embeds = nn.Embedding(4,
                                       4,
                                       max_norm=True,
                                       scale_grad_by_freq=True)

        # 		self.icd_embeds = nn.Embedding(
        # 			self.icd_dim, self.icd_embed_dim, max_norm=None, scale_grad_by_freq=True
        # 		)

        self.sex_age = nn.Bilinear(2, 4, self.dim_agg)
        self.prelu_sex_age = nn.PReLU()

        # 		self.L = nn.Linear(self.dim_agg, self.dim_agg * 8)
        self.L1 = nn.Linear(self.dim_agg, self.dim_agg)
        self.L2 = nn.Linear(self.dim_agg, self.dim_agg)
        self.L3 = nn.Linear(self.dim_agg, self.dim_agg)
        self.encorder_atc = EncoderRNN(
            input_size=self.dim_agg,
            hidden_size=self.dim_agg,
        )
        self.encorder_pro = EncoderRNN(
            input_size=self.dim_agg,
            hidden_size=self.dim_agg,
        )
        self.encorder_icd = EncoderRNN(
            input_size=self.dim_agg,
            hidden_size=self.dim_agg,
        )
        self.L_16to8 = nn.Linear(self.dim_agg * 12, self.dim_agg * 1)
        self.Sequential_1 = nn.Sequential(
            OrderedDict([('linear_1',
                          nn.Linear(self.dim_agg * 1, self.dim_agg * 1)),
                         ('prelu_1', nn.PReLU())]))

        self.Sequential_2 = nn.Sequential(
            OrderedDict([('linear_1',
                          nn.Linear(self.dim_agg * 2, self.dim_agg * 1)),
                         ('prelu_1', nn.PReLU())]))

        self.Sequential_3 = nn.Sequential(
            OrderedDict([('linear_1',
                          nn.Linear(self.dim_agg * 4, self.dim_agg * 1)),
                         ('prelu_1', nn.PReLU())]))
        self.L8to1cd = nn.Linear(self.dim_agg * 8, icd_dim)
        # 		self.PReLU = nn.PReLU()
        self.Softmax = nn.Softmax(dim=1)


#         self.decorder = AttnDecoderRNN(
# 			hidden_size=self.dim_agg,
# 			output_size=self.icd_embed_dim,
# 			dropout_p=0.1,
# 			icd_embeds=self.icd_embeds
# 		)

    def forward(self, pro, atc, age, sex, icd, len_atc, len_pro, len_icd):

        B = pro.shape[0]

        pro = self.pro_embeds(pro)
        atc = self.atc_embeds(atc)
        icd = self.icd_embeds(icd)
        # 		print(age, sex)
        age = self.age_embeds(age)
        sex = self.sex_embeds(sex)

        h = self.sex_age(sex, age)
        h = self.prelu_sex_age(h)
        # 		h = (
        # 			self.L(h)
        # 			.view(B, 8, self.dim_agg)
        # 			.permute(1, 0, 2)
        # 		)

        tmp = torch.stack(
            tuple(self.atc_embeds(self.atc_start) for i in range(B)),
            dim=0,
        )
        atc = torch.cat((tmp, atc), dim=1)
        atc = self.L1(torch.mul(atc, h.unsqueeze(1)))
        tmp = torch.stack(
            tuple(self.pro_embeds(self.pro_start) for i in range(B)),
            dim=0,
        )
        pro = torch.cat((tmp, pro), dim=1)
        pro = self.L2(torch.mul(pro, h.unsqueeze(1)))
        tmp = torch.stack(
            tuple(self.icd_embeds(self.icd_start) for i in range(B)),
            dim=0,
        )
        icd = torch.cat((tmp, icd), dim=1)
        icd = self.L3(torch.mul(icd, h.unsqueeze(1)))
        atc = pack_padded_sequence(
            atc,
            len_atc + 1,
            batch_first=True,
            enforce_sorted=False,
        )
        atc, _ = self.encorder_atc(atc, torch.rand(1, B, self.dim_agg).cuda())
        # 		print(atc.data.shape)
        atc, _ = pad_packed_sequence(atc, batch_first=True)

        pro = pack_padded_sequence(
            pro,
            len_pro + 1,
            batch_first=True,
            enforce_sorted=False,
        )
        pro, _ = self.encorder_pro(pro, torch.rand(1, B, self.dim_agg).cuda())
        pro, _ = pad_packed_sequence(pro, batch_first=True)
        icd = pack_padded_sequence(
            icd,
            len_icd + 1,
            batch_first=True,
            enforce_sorted=False,
        )
        icd, _ = self.encorder_icd(icd, torch.rand(1, B, self.dim_agg).cuda())
        icd, _ = pad_packed_sequence(icd, batch_first=True)
        atc = torch.cat(
            (atc.sum(1) / (atc != 0).sum(1), torch.sum(atc, dim=1),
             torch.max(atc, dim=1)[0], torch.std(atc, dim=1, unbiased=True)),
            1)
        pro = torch.cat(
            (pro.sum(1) / (pro != 0).sum(1), torch.sum(pro, dim=1),
             torch.max(pro, dim=1)[0], torch.std(pro, dim=1, unbiased=True)),
            1)
        icd = torch.cat(
            (icd.sum(1) / (icd != 0).sum(1), torch.sum(icd, dim=1),
             torch.max(icd, dim=1)[0], torch.std(icd, dim=1, unbiased=True)),
            1)
        x = torch.cat((atc, pro, icd), dim=1)
        x = self.L_16to8(x)
        x1 = torch.cat((self.Sequential_1(x), x), dim=1)
        x2 = torch.cat((self.Sequential_2(x1), x, x1), dim=1)
        x3 = torch.cat((self.Sequential_3(x2), x, x1, x2), dim=1)
        x = self.Softmax(self.L8to1cd(x3))
        return x


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, output, hidden):
        output, hidden = self.gru(output, hidden)
        return output, hidden


def arg_parse():
    parser = argparse.ArgumentParser(description="DL")

    # Datasets parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="root path to data directory",
    )
    parser.add_argument(
        "--workers",
        default=6,
        type=int,
        help="number of data loading workers (default: 4)",
    )

    # training parameters
    parser.add_argument(
        "--gpu",
        default=1,
        type=int,
        help="In homework, please always set to 0",
    )
    parser.add_argument(
        "--epoch",
        default=25,
        type=int,
        help="num of validation iterations",
    )
    parser.add_argument(
        "--val_epoch",
        default=-1,
        type=int,
        help="num of validation iterations",
    )
    parser.add_argument(
        "--train_batch",
        default=32,
        type=int,
        help="train batch size",
    )
    parser.add_argument(
        "--accumulation_steps",
        default=1,
        type=int,
        help="accumulation_steps size",
    )
    parser.add_argument(
        "--test_batch",
        default=64,
        type=int,
        help="test batch size",
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0002,
        type=float,
        help="initial learning rate",
    )

    # resume trained model
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to the trained model",
    )
    # others
    parser.add_argument("--save_dir", type=str, default="model_AC_prime_sex")
    parser.add_argument("--random_seed", type=int, default=999)

    args = parser.parse_args()

    return args


def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    print("===> set device ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("===> loading dicts ...")

    with open(r".//icd_dict_AC.pkl", "rb") as f:
        icd_dict = pkl.load(f)
    with open(r".//icd_dict2_AC.pkl", "rb") as f:
        icd_dict2 = pkl.load(f)
    with open(r".//pro_AC.pkl", "rb") as f:
        pro_dict = pkl.load(f)
    with open(r".//atc_AC.pkl", "rb") as f:
        atc_dict = pkl.load(f)
    with open(r".//raw_AC.pkl", "rb") as f:
        data = pkl.load(f).head(1000)

# 	dataset = MyDataset(data, icd_dict, atc_dict, pro_dict)
# 	train_size = int(0.8 * len(dataset))
# 	test_size = len(dataset) - train_size
# 	train_dataset, test_dataset = torch.utils.data.random_split(
# 		dataset, [train_size, test_size])

    ids = data.id.drop_duplicates()
    x = np.random.permutation(len(ids))
    test = (x >= 0.8 * len(ids)).tolist()
    train = (x < 0.8 * len(ids)).tolist()
    train_id = ids.loc[train]
    test_id = ids.loc[test]
    print("===> build dataset ...")
    train_dataset = MyDataset(data.merge(train_id, on="id"), icd_dict,
                              icd_dict2, atc_dict, pro_dict)
    test_dataset = MyDataset(data.merge(test_id, on="id"), icd_dict, icd_dict2,
                             atc_dict, pro_dict)

    with open(r"./train_dataset.pkl", "wb") as f:
        pkl.dump(train_dataset, f)
    with open(r"./test_dataset.pkl", "wb") as f:
        pkl.dump(test_dataset, f)

# 	with open(r'./train_dataset.pkl', 'rb') as f:
# 		train_dataset = pkl.load(f)
# 	with open(r'./test_dataset.pkl', 'rb') as f:
# 		test_dataset = pkl.load(f)

    print("===> build dataloader ...")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch,
        num_workers=args.workers,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch,
        num_workers=args.workers,
        shuffle=False,
    )

    print("===> basic setting ...")
    model = Model(
        atc_start=torch.tensor(atc_dict.transform(['start'])).to(device),
        pro_start=torch.tensor(pro_dict.transform(['start'])).to(device),
        icd_start=torch.tensor(icd_dict2.transform(['start'])).to(device),
        atc_dim=len(atc_dict.classes_),
        pro_dim=len(pro_dict.classes_),
        icd_dim=len(icd_dict.classes_),
    ).to(device)
    # model.load_state_dict(torch.load('unet.pt'))
    # 	criteria1, criteria2 = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), reduction='none'), nn.MSELoss(reduction='mean')
    criteria1, criteria2 = nn.BCELoss(), nn.MSELoss()
    threshold = 0.5
    best_f1 = 0.
    #    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = AdaBelief(model.parameters(), lr=args.lr)
    print("===> record list setting ...")
    epoch_loss = []
    batch_loss = []
    accuracy = []
    f1 = []
    precision = []
    coverage = []
    recall = []
    labelrankingloss = []
    val_pred2 = []
    a = []
    f = []
    c = []
    p = []
    r = []
    l = []
    for epoch in range(1, args.epoch + 1):
        print("===> set training mode ...")

        model.train()
        model.zero_grad()
        batch_loss.clear()
        for idx, (pro, atc, age, sex, icd_in, icd_out, len_atc, len_pro,
                  len_icd_in, len_icd_out) in enumerate(train_loader):

            train_info = "Epoch: [{0}][{1}/{2}]".format(
                epoch, idx + 1, len(train_loader))
            optimizer.zero_grad()
            pro = pro.to(device, dtype=torch.long)
            atc = atc.to(device, dtype=torch.long)
            age = age.to(device, dtype=torch.long)
            sex = sex.to(device, dtype=torch.long)
            icd_in = icd_in.to(device, dtype=torch.long)
            icd_out = icd_out.to(device, dtype=torch.float).squeeze()
            len_atc = len_atc.to(dtype=torch.long)
            len_pro = len_pro.to(dtype=torch.long)
            len_icd_in = len_icd_in.to(dtype=torch.long)
            len_icd_out = len_icd_out.to(dtype=torch.long)

            pred = model(pro, atc, age, sex, icd_in, len_atc, len_pro,
                         len_icd_in)

            loss = criteria1(
                pred,
                icd_out)  #(criteria1(pred, icd) + criteria2(pred, icd))/2
            loss.backward()

            # torch.cuda.empty_cache()
            # print('loss = '+ str(loss.item()), end="\r")

            optimizer.step()

            batch_loss += [loss.item()]

            train_info += "Loss: {:.12f}".format(loss.item())
            print(train_info, end="\r")

        epoch_loss += [sum(batch_loss) / len(batch_loss)]
        train_info = "Epoch: [{0}], Loss: {1}".format(epoch, epoch_loss[-1])
        print(train_info)
        if epoch % args.val_epoch == 0:
            model.eval()
            a.clear()
            f.clear()
            c.clear()
            p.clear()
            r.clear()
            l.clear()
            with torch.no_grad():
                for idx, (pro, atc, age, sex, icd_in, icd_out, len_atc,
                          len_pro, len_icd_in,
                          len_icd_out) in enumerate(test_loader):
                    print(idx, end="\r")
                    pro = pro.to(device, dtype=torch.long)
                    atc = atc.to(device, dtype=torch.long)
                    age = age.to(device, dtype=torch.long)
                    sex = sex.to(device, dtype=torch.long)
                    icd_in = icd_in.to(device, dtype=torch.long)
                    icd_out = icd_out.to(device, dtype=torch.float).squeeze()
                    len_atc = len_atc.to(dtype=torch.long)
                    len_pro = len_pro.to(dtype=torch.long)
                    len_icd_in = len_icd_in.to(dtype=torch.long)
                    len_icd_out = len_icd_out.to(dtype=torch.long)
                    pred = model(pro, atc, age, sex, icd_in, len_atc, len_pro,
                                 len_icd_in)
                    val_pred = (pred.cpu().numpy() > 0.5)  # .tolist()
                    val_pred2 = pred.cpu().numpy()  # .tolist()
                    val_label = icd_out.cpu().numpy()  # .tolist()
                    a += [
                        label_ranking_average_precision_score(
                            val_label, val_pred2)
                    ]
                    f += [
                        f1_score(val_label,
                                 val_pred,
                                 average='micro',
                                 zero_division=0)
                    ]
                    c += [coverage_error(val_label, val_pred2)]
                    p += [
                        precision_score(val_label,
                                        val_pred,
                                        average='micro',
                                        zero_division=0)
                    ]
                    r += [
                        recall_score(val_label,
                                     val_pred,
                                     average='micro',
                                     zero_division=0)
                    ]
                    l += [label_ranking_loss(val_label, val_pred2)]


# 				val_pred = val_pred > 0.5
                accuracy += [sum(a) / len(a)]
                coverage += [sum(c) / len(c)]
                f1 += [sum(f) / len(f)]
                precision += [sum(p) / len(p)]
                recall += [sum(r) / len(r)]
                labelrankingloss += [sum(l) / len(l)]

                print(
                    'Epoch: [{}] label_ranking_average_precision_score:{}, f1_score:{}, precision:{}, coverage_error:{}, recall:{}, label_ranking_loss:{}'
                    .format(epoch, accuracy[-1], f1[-1], precision[-1],
                            coverage[-1], recall[-1], labelrankingloss[-1]))
                if f1[-1] > best_f1:
                    save_model(model,
                               os.path.join(args.save_dir, 'model_best.pkl'))
                    best_f1 = f1[-1]

        save_model(
            model,
            os.path.join(args.save_dir,
                         'model' + str(epoch) + '_f1_{}.pkl'.format(f1[-1])))
        result = pd.DataFrame(
            data={
                'label_ranking_average_precision_score': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'label_ranking_loss': labelrankingloss
            })
        ax = result.plot()
        fig = ax.get_figure()
        fig.savefig(os.path.join(args.save_dir, 'result.jpeg'))
        plt.close()

        coverageError = pd.DataFrame(data={'coverage_error': coverage})
        ax = coverageError.plot()
        fig = ax.get_figure()
        fig.savefig(os.path.join(args.save_dir, 'coverage_error.jpeg'))
        plt.close()

        result_loss = pd.DataFrame(data={'BCE loss': epoch_loss})
        ax = result_loss.plot()
        fig = ax.get_figure()
        fig.savefig(os.path.join(args.save_dir, 'loss.jpeg'))
        plt.close()

        coverageError.to_csv(os.path.join(args.save_dir, 'coverage_error.csv'))
        result.to_csv(os.path.join(args.save_dir, 'result.csv'))
        result_loss.to_csv(os.path.join(args.save_dir, 'BCE loss.csv'))


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


class MyDataset(Dataset):

    def __init__(self, data, icd_dict, icd_dict2, atc_dict, pro_dict):
        self.data = data

        self.icd_dict = icd_dict
        self.icd_dict2 = icd_dict2
        self.atc_dict = atc_dict
        self.pro_dict = pro_dict

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        #         ID_SEX	AGE	FUNC_TYPE	ICD9CM_CODE	ORDER_CODE
        icd_out = self.data["ICD9CM_CODE_OUTPUT"].iloc[idx]
        icd_in = self.data["ICD9CM_CODE"].iloc[idx]
        pro = self.data["PRO"].iloc[idx]
        atc = self.data["ATC"].iloc[idx]
        sex = self.data["ID_SEX"].iloc[idx]
        age = self.data["AGE"].iloc[idx]
        # 		print(type(icd_out))
        # 		print(icd_out)
        icd_out = self.icd_dict.transform([[icd_out]])
        # 		print(type(icd_out))
        # 		print(icd_out)
        len_atc = len(atc)
        len_pro = len(pro)
        len_icd_in = len(icd_in)
        len_icd_out = len(icd_out)

        icd_out = torch.tensor(icd_out, dtype=torch.long)

        icd_in = self.icd_dict2.transform(icd_in)

        icd_in = F.pad(
            torch.Tensor(icd_in)[torch.randperm(len(icd_in))],
            pad=(0, 4 - len(icd_in)),
            value=self.icd_dict2.transform(["null"]).item(),
        )

        pro = self.pro_dict.transform(pro).tolist()
        # 		print(order)
        pro = F.pad(
            torch.Tensor(pro)[torch.randperm(len(pro))],
            pad=(0, 2048 - len(pro)),
            value=self.pro_dict.transform(["null"]).item(),
        )

        atc = self.atc_dict.transform(atc).tolist()
        # 		print(order)
        atc = F.pad(
            torch.Tensor(atc)[torch.randperm(len(atc))],
            pad=(0, 1024 - len(atc)),
            value=self.atc_dict.transform(["null"]).item(),
        )
        return pro, atc, age, sex, icd_in, icd_out, len_atc, len_pro, len_icd_in, len_icd_out


if __name__ == "__main__":
    args = arg_parse()

    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
