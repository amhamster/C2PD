# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import loss
import torch
import utility
from option import args
from Trainer import Trainer

utility.init_state()
utility.make_checkpoint_dir(args.file_name)

loss = loss.Loss()
device = torch.device('cpu' if args.cpu else 'cuda')

from model import C2PD

model = C2PD().to(device)

if not args.cpu:
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))

pre = True    
    
if pre:
    if args.scale == 4:
        load_name = './pre_trained/x4.pth'
    elif args.scale == 8:
        load_name = './pre_trained/x8.pth'
    elif args.scale == 16:
        load_name = './pre_trained/x16.pth'
    elif args.scale == 32:
        load_name = './pre_trained/x32.pth'

    print(load_name)

    checkpoint = torch.load(load_name)
    model.load_state_dict(checkpoint)    
    
train_process = Trainer(args=args, my_model=model, my_loss=loss)

train_process.train()