"""
Author: Arun Balajee Vasudevan
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import torch.nn.functional as F
from apex import amp

from config import cfg, assert_and_infer_cfg
from utils.eval_misc_fullSeg_noSkip import AverageMeter, prep_experiment, evaluate_eval, fast_hist1, warpgrid, evaluate_eval_for_inference
import datasets
import loss_fullSeg as loss
import network
import optimizer_two5 as optimizer
import cv2
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--arch', type=str, default='network.deepv3_audioBG_noBG_diffmask.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='OmniAudio_audioBG_fullSeg_Logmag_diffmask',
                    help='cityscapes, mapillary, camvid, kitti, OmniAudio')

parser.add_argument('--cv', type=int, default=None,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=True,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=True,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adam', action='store_true', default=True)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default="/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/audio/semanticPred/logs/ckpt/default/Omni-network.deepv3_audioBG_noBG_diffmask.DeepWV3Plus/models_NormAudio_noskip_logmap/SOP_epoch_4.pth")
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ and args.apex:
    args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

if args.apex:
    # Check that we are running with cuda as distributed is only supported for cuda.
    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)
    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

def spectrogram_for_SOP(mag_img):
    B = mag_img.shape[0]
    T = mag_img.shape[3]
    grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).cuda()
    mag_img = F.grid_sample(mag_img, grid_warp)
    log_mag_img = torch.log(mag_img).detach()
    return log_mag_img

def main():
    """
    Main Function
    """

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = loss.get_loss(args)
    net = network.get_net(args, criterion)
    optim, scheduler = optimizer.get_optimizer(args, net)

    if args.fp16:
        net, optim = amp.initialize(net, optim, opt_level="O1")

    #net = network.warp_network_in_dataparallel(net, args.apex)
    if args.snapshot:
        optimizer.load_weights(net, optim,
                               args.snapshot, args.restore_optimizer)

    torch.cuda.empty_cache()
    # Main Loop
    for epoch in range(1):#args.start_epoch, args.max_epoch):
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)

        scheduler.step()
        #train(train_loader, net, optim, epoch, writer)
        if args.apex:
            train_loader.sampler.set_epoch(epoch + 1)
        validate(val_loader, net, criterion_val,optim, epoch, writer)
        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                if args.apex:
                    train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()


def train(train_loader, net, optim, curr_epoch, writer):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_main_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        # inputs = (2,3,713,713)
        # gts    = (2,713,713)
        in_imgs, gts,sem_mask, in_aud1, in_aud6, _img_name = data

        batch_pixel_size = in_imgs.size(0) * in_imgs.size(2) * in_imgs.size(3)
        #print(in_imgs.size())
        in_imgs, gts, in_aud1, in_aud6 = in_imgs.type(torch.FloatTensor).cuda(), gts.cuda(), in_aud1.cuda(), in_aud6.cuda()
        #inputs = spectrogram_for_SOP(inputs)
        #print(inputs.shape)

        optim.zero_grad()
        #print(inputs[0,0,20:30,30:40])

        main_loss = net(in_aud1, in_aud6, gts=gts)

        if args.apex:
            log_main_loss = main_loss.clone().detach_()
            torch.distributed.all_reduce(log_main_loss, torch.distributed.ReduceOp.SUM)
            log_main_loss = log_main_loss / args.world_size
        else:
            main_loss = main_loss.mean()
            log_main_loss = main_loss.clone().detach_()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)
        if args.fp16:  # and 0:
            with amp.scale_loss(main_loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            main_loss.backward()
        clip=1
        torch.nn.utils.clip_grad_norm_(net.parameters(),clip)
        optim.step()

        curr_iter += 1

        if args.local_rank == 0:
            msg = '[epoch {}], [iter {} / {}], [train main loss {:0.6f}], [lr {:0.6f}]'.format(
                curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
                optim.param_groups[-1]['lr'])

            logging.info(msg)

            # Log tensorboard metrics for each iteration of the training phase
            writer.add_scalar('training/loss', (train_main_loss.val),
                              curr_iter)
            writer.add_scalar('training/lr', optim.param_groups[-1]['lr'],
                              curr_iter)

        if i > 5 and args.test_mode:
            return


def validate(val_loader, net, criterion, optim, curr_epoch, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        in_imgs, gt_image, sem_mask, in_aud1, in_aud6, img_names = data
        assert len(in_imgs.size()) == 4 and len(gt_image.size()) == 3
        #assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = in_imgs.size(0) * in_imgs.size(2) * in_imgs.size(3)
        in_imgs, gt_cuda,sem_mask, in_aud1, in_aud6 = in_imgs.type(torch.FloatTensor).cuda(), gt_image.cuda(),sem_mask.cuda(), in_aud1.cuda(), in_aud6.cuda()
        #inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(in_aud1, in_aud6, gts=gt_cuda)  # output = (1, 19, 713, 713)

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == args.dataset_cls.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()
        #### Saving the prediction ####
        '''
        save_dir="/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/audio/semanticPred/Results/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pred13 = output.data[0,14,:,:].cpu().numpy();
        #pred13 = (pred13-pred13.min())/(pred13.max()-pred13.min());pred13=np.log(pred13);
        pred13 = (pred13-pred13.min())*255/(pred13.max()-pred13.min());#print(pred13)
        cv2.imwrite(os.path.join(save_dir, "%06d"%val_idx+'.png'),pred13) 
        '''
        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx%10==0:
            dump_images.append([gt_image, predictions, img_names])
        for k in range(19):
            if k in [0,1,2,3,4,5,6,7,8,9,10,12]:    predictions[predictions == k] = 0;gt_image[gt_image == k] = 0;
            #else:    mask_copy[mask == k] = v
        #print(predictions)

        iou_acc += fast_hist1(predictions.numpy().flatten(), gt_image.numpy().flatten(), sem_mask.cpu().numpy().flatten(), args.dataset_cls.num_classes)
        del output, val_idx, data

    if args.apex:
        iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
        torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
        iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, val_loss, iou_acc, dump_images,
                      writer, curr_epoch, args.dataset_cls)
        #acc, acc_cls, mean_iu, fwavacc = evaluate_eval_for_inference(iou_acc, args.dataset_cls)
        #print(mean_iu)

    return val_loss.avg


if __name__ == '__main__':
    main()
