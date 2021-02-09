"""
Author: Arun Balajee Vasudevan
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if args.sgd:
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.adam:
        amsgrad = False
        if args.amsgrad:
            amsgrad = True
        #print(net)  
        optimizer = optim.Adam([{'params':net.audionet_convlayer1.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer2.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer3.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer4.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer5.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer1.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer2.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer3.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer4.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer5.parameters(),'lr':args.lr*5},{'params':net.conv1x1.parameters()},{'params':net.final.parameters()},{'params':net.final_depth.parameters()},{'params':net.bot_aspp.parameters()},{'params':net.bot_depthaspp.parameters()},{'params':net.aspp.parameters()},{'params':net.depthaspp.parameters()},{'params':net.bot_fine.parameters()},{'params':net.bot_multiaud.parameters()},{'params':net.bot_aud1.parameters()}],
        #optimizer = optim.Adam([{'params':net.audionet_convlayer1.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer2.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer3.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer4.parameters(),'lr':args.lr*5},{'params':net.audionet_convlayer5.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer1.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer2.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer3.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer4.parameters(),'lr':args.lr*5},{'params':net.audionet_upconvlayer5.parameters(),'lr':args.lr*5},{'params':net.conv1x1.parameters()},{'params':net.final.parameters()},{'params':net.bot_aspp.parameters()},{'params':net.bot_fine.parameters()},{'params':net.bot_multiaud.parameters()},{'params':net.bot_aud1.parameters()},{'params':net.aspp.parameters()},{'params':net.mod7.parameters()},{'params':net.mod6.parameters()},{'params':net.mod5.parameters()},{'params':net.mod4.parameters()}],
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=amsgrad
                               )
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
             math.pow(1 - epoch / args.max_epoch,
                      args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
