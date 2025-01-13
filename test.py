# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import torch
from option import args
from Tester import Tester

device = torch.device('cpu' if args.cpu else 'cuda')

from model import C2PD
model = C2PD().to(device)

if not args.cpu:
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))

device_id = torch.cuda.current_device()

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

train_process = Tester(args=args, my_model=model)

train_process.test()