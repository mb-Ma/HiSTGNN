
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.AGCRN import AGCRN as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters



#*************************************************************************#
Mode = 'train'
DEBUG = 'True'
DATASET = 'USA'      #PEMSD4 or PEMSD8
DEVICE = 'cuda:0'
MODEL = 'AGCRN'

#get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

from lib.metrics import MAE_torch


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=0.1, type=float)
args.add_argument('--test_ratio', default=0.2, type=float)
args.add_argument('--lag', default=48, type=int)
args.add_argument('--horizon', default=24, type=int)
args.add_argument('--num_nodes', default=4, type=int)
args.add_argument('--tod', default=False, type=eval)
args.add_argument('--normalizer', default='max-min', type=str)
args.add_argument('--column_wise', default=False, type=eval)
args.add_argument('--default_graph', default=False, type=eval)
#model
args.add_argument('--input_dim', default=2, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--embed_dim', default=40, type=int)
args.add_argument('--rnn_units', default=32, type=int)
args.add_argument('--num_layers', default=2, type=int)
args.add_argument('--cheb_k', default=2, type=int)
#train
args.add_argument('--loss_func', default='mask_mae', type=str)
args.add_argument('--seed', default=10, type=int)
args.add_argument('--batch_size', default=512, type=int)
args.add_argument('--epochs', default=50, type=int)
args.add_argument('--lr_init', default=0.001, type=float)
args.add_argument('--lr_decay', default=False, type=eval)
args.add_argument('--lr_decay_rate', default=0.3, type=float)
args.add_argument('--lr_decay_step', default='5,20,40,70', type=str)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=10, type=int)
args.add_argument('--grad_norm', default=False, type=eval)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=True, type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=None, type=eval)
args.add_argument('--mape_thresh', default=0., type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--plot', default=False, type=eval)
args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(None, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)

if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('../pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
