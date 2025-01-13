# -*- coding: utf-8 -*-

import os
import argparse
parser = argparse.ArgumentParser(description='Guided Depth Super-Resolution')

parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--cuda_name', type=int, default=0)

parser.add_argument('--data_augment', type=bool, default=True)
parser.add_argument('--dataset_name', type=str, default='NYU')
parser.add_argument('--test_set', type=str, default='Middlebury+Lu+test+RGBDD') 

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--lr_decay_rate', type=float, default=0.5)
parser.add_argument('--lr_decay_epochs', type=str, default='360')

parser.add_argument('--scale', type=int, default=8)
parser.add_argument('--real', type=bool, default=False)

parser.add_argument('--num_epochs', type=int, default=501)
parser.add_argument('--test_epochs', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=1)

parser.add_argument('--model_name', type=str, default='C2PD')
parser.add_argument('--file_name', type=str, default='')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
    elif vars(args)[arg] == 'None':
        vars(args)[arg] = None

args.lr_decay_epochs = [int(num) for num in args.lr_decay_epochs.split('_')]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if args.real:
    args.scale = 32

if len(args.file_name) == 0:
    args.file_name = args.model_name + '_' + str(args.scale)

print(args)
