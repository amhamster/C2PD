# -*- coding: utf-8 -*-

from torch.optim.lr_scheduler import MultiStepLR


def get_scheduler(optimizer, n_iter_per_epoch, args):

    lr_decay_epochs = args.lr_decay_epochs
    scheduler = MultiStepLR(
        optimizer=optimizer,
        gamma=args.lr_decay_rate,
        milestones=[m * n_iter_per_epoch for m in lr_decay_epochs])

    return scheduler