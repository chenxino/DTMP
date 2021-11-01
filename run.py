import os
import sys

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import copy

from datetime import datetime

from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch

from utils import normalized_network, get_adjacency_matrix
from train import Trainer
from model import Anet

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=True, type=eval)
    parser.add_argument('--dataset', default='PEMSD4', type=str)
    parser.add_argument('--model', default='Anet', type=str)
    # train
    parser.add_argument('--loss_func', default='mae', type=str)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--epochs',type=int,default=200,help='')
    parser.add_argument('--lr_init', default=0.003, type=float)
    parser.add_argument('--lr_decay', default=False, type=eval)
    parser.add_argument('--lr_decay_rate', default=0.3, type=float)
    parser.add_argument('--lr_decay_step', default='5,20,40,70', type=str)
    parser.add_argument('--early_stop', default=True, type=eval)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--grad_norm', default=False, type=eval)
    parser.add_argument('--max_grad_norm', default=5, type=int)
    parser.add_argument('--real_value', default=True, type=eval, help = 'use real value for loss calculation')
    # test
    parser.add_argument('--mae_thresh', default=None, type=eval)
    parser.add_argument('--mape_thresh', default=0., type=float)
    # log
    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=20, type=int)
    parser.add_argument('--plot', default=False, type=eval)
    # model
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--seq_length',type=int,default=12,help='')
    parser.add_argument('--nhid',type=int,default=32,help='')
    parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    parser.add_argument('--kernel_size',type=int,default=2,help='')
    parser.add_argument('--blocks',type=int,default=2,help='')
    parser.add_argument('--layers',type=int,default=3,help='')
    parser.add_argument('--e_dim',type=int,default=10,help='node embedded dimensions')
    parser.add_argument('--kernel_size_Agcn',type=int,default=2,help='')

    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    # data
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--num_nodes', default=307, type=int)
    parser.add_argument('--tod', default=False, type=eval)
    parser.add_argument('--normalizer', default='std', type=str)
    parser.add_argument('--column_wise', default=False, type=eval)
    parser.add_argument('--default_graph', default=False, type=eval)

    parser.add_argument('--input_dim', default=1, type=int)  # 注意get_dataload里面用了这个
    parser.add_argument('--output_dim', default=1, type=int)

    args = parser.parse_args()

    init_seed(args.seed)

    device = torch.device(args.device)

    train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                      normalizer=args.normalizer,
                                      tod=args.tod, dow=False,
                                      weather=False, single=False)

    if args.default_graph:
        adjinit, distance = get_adjacency_matrix('./data/PeMSD4/distance.csv', args.num_nodes)
    else:
        adjinit = None # 无预定义矩阵
    
    if args.model == 'Anet':
        model = Anet(device, args.num_nodes, args.dropout, aptinit=adjinit, 
        in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, 
        dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16, kernel_size=args.kernel_size,blocks=args.blocks,layers=args.layers, e_dim=args.e_dim, kernel_size_Agcn = args.kernel_size_Agcn)
    else:
        raise ValueError
    model = model.to(device)
    # print_model_parameters(model, only_num=False)
    
    #init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,
                weight_decay=args.weight_decay)

    #learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)
    #config log path
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.join('./log')
    log_dir = os.path.join(current_dir,'experiments', args.dataset, '{}_{}_{}_{}_{}'.format(
            current_time, args.model, args.lr_init, args.input_dim, args.num_nodes))
    args.log_dir = log_dir

    #start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                    args, lr_scheduler=lr_scheduler)
    trainer.train()
    
