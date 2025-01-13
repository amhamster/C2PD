# -*- coding: utf-8 -*-
import tqdm
import time
import torch
import utility
import torchnet as tnt
from data import get_dataloader
from prettytable import PrettyTable
from lr_scheduler import get_scheduler
from torch.nn.functional import interpolate

class Trainer():
    def __init__(self, args, my_model, my_loss):
        self.args = args
        self.loss = my_loss
        self.model = my_model
        self.start_time = time.time()
        self.epoch_num = self.step = 0
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.optimizer = utility.make_optimizer(self.args, self.model)
        self.loader_train = get_dataloader(args=self.args, attr='train').loader_train
        self.scheduler = get_scheduler(self.optimizer, n_iter_per_epoch=len(self.loader_train), args=args)

    def train(self):
        self.model.train()
        train_loss = tnt.meter.AverageValueMeter()
        train_rmse = tnt.meter.AverageValueMeter()
        for epoch_num in range(self.epoch_num, self.args.num_epochs):
            self.epoch_num = epoch_num

            p_bar = tqdm.tqdm(self.loader_train)

            for _, sample in enumerate(p_bar):
                self.step += 1
                self.optimizer.zero_grad()
                lr_img, gt_img, rgb_img, min_value, max_value = sample['lr_img'].to(self.device), sample['gt_img'].to(self.device), sample[
                    'rgb_img'].to(self.device), sample['min_value'].to(self.device), sample['max_value'].to(self.device)

                lr_up = interpolate(lr_img, scale_factor=self.args.scale, mode='bicubic', align_corners=False)
                
                out_img = self.model(rgb=rgb_img, lr_up=lr_up, train=True)

                loss = self.loss(out_img, gt_img)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                rmse, _ = utility.root_mean_sqrt_error(im_pred=out_img, im_true=gt_img, min_value=min_value, max_value=max_value, border=0, is_train=True)
                train_rmse.add(rmse)
                train_loss.add(loss.item())
                p_bar.set_description('===> Epoch: {}'.format(str(self.epoch_num)).zfill(3))
                p_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'], RMSE=rmse)

            print('===> Epoch: {}, Step: {:<5d}, {:<5s}_loss: {:.4f}, {:<5s}_rmse: {:.4f}, time_spend: {}, lr: {}'
                  .format(self.epoch_num, self.step, 'train', 10000 * train_loss.value()[0], 'train',
                          train_rmse.value()[0], utility.time_since(self.start_time),
                          self.optimizer.param_groups[0]['lr']))

            train_rmse.reset()
            train_loss.reset()
            if self.epoch_num % self.args.test_epochs == 0:
                self.test()

    def test_model(self, attr, border, mul_ratio, is_train):
        self.model.eval()
        test_loader = get_dataloader(self.args, attr).data_loader

        sum_times = 0
        rmse_list = []

        test_rmse = tnt.meter.AverageValueMeter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _, sample in enumerate(test_loader):
            lr_img, gt_img, rgb_img, min_value, max_value = sample['lr_img'].to(self.device), sample['gt_img'].to(
                self.device), sample['rgb_img'].to(self.device), sample['min_value'].to(self.device), sample['max_value'].to(self.device)
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
            test_set = self.args.test_set.split('+')
            for test_name in test_set:
                mul_ratio = 1
                if test_name == 'test':
                    mul_ratio = 100
                if test_name == 'RGBDD':
                    mul_ratio = 0.1

                test_rmse, test_time, rmse_list = self.test_model(test_name, 6, mul_ratio, is_train=False)
                test_data_name.append(test_name)
                test_data_rmse.append(round(test_rmse, 4))
        
        torch.save(self.model.state_dict(),
                   './checkpoints/{}/epo_{}.pth'.format(self.args.file_name, str(self.epoch_num)))

        table = PrettyTable(test_data_name)
        table.add_row(test_data_rmse)
        print(table)
