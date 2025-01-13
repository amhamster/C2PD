# -*- coding: utf-8 -*-

import torch
import torchnet as tnt
from data import get_dataloader
from prettytable import PrettyTable
from torch.nn.functional import interpolate
import utility

class Tester():
    def __init__(self, args, my_model):
        self.args = args
        self.model = my_model
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')

        self.test_name = 'None'

    def test_model(self, attr, border, mul_ratio, is_train):
        self.model.eval()
        test_loader = get_dataloader(self.args, attr).data_loader

        sum_times = 0
        rmse_list = []

        test_rmse = tnt.meter.AverageValueMeter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _, sample in enumerate(test_loader):
            lr_img, gt_img, rgb_img, min_value, max_value = self.prepare(sample['lr_img'], sample['gt_img'], sample['rgb_img'], sample['min_value'], sample['max_value'])
            lr_up = interpolate(lr_img, scale_factor=self.args.scale, mode='bicubic', align_corners=False)
            start.record()
 
            out = self.model(rgb=rgb_img, lr_up=lr_up)

            end.record()
            torch.cuda.synchronize()
            sum_times += start.elapsed_time(end)
            
            rmse, im_pred = utility.root_mean_sqrt_error(im_pred=out, im_true=gt_img, min_value=min_value, max_value=max_value, border=border,
                                                         mul_ratio=mul_ratio, is_train=is_train)

            rmse_list.append(rmse)
            test_rmse.add(rmse)
        return test_rmse.value()[0], round(sum_times / 1000, 5), rmse_list

    def test(self):
        print("===> Testing model...")
        test_data_name = []
        test_data_rmse = []
        with torch.no_grad():
            if self.args.real:
                test_set = ['RGBDD']
            else:
                test_set = self.args.test_set.split('+')
            for test_name in test_set:
                self.test_name = test_name
                mul_ratio = 1
                if test_name == 'test':
                    mul_ratio = 100
                elif test_name == 'RGBDD':
                    mul_ratio = 0.1

                test_rmse, test_time, rmse_list = self.test_model(test_name, 6, mul_ratio, is_train=False)

                test_data_name.append(test_name)
                test_data_rmse.append(round(test_rmse, 4))

        table = PrettyTable(test_data_name)
        table.add_row(test_data_rmse)
        print(table)

    def prepare(self, *args):
        def _prepare(tensor):
            return tensor.to(self.device).contiguous()

        return [_prepare(a) for a in args]
