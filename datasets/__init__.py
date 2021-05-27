"""
# Code Adapted from:
# https://github.com/NVIDIA/semantic-segmentation/
#
# Copyright 2020 Nvidia Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
"""
Dataset setup and loaders
"""

from datasets import OmniAudio_noBG_Paralleltask
from datasets import OmniAudio_noBG_Paralleltask_depth
from datasets import OmniAudio_noBG_Paralleltask_depth_noSeman
import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader


def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """
    if args.dataset == 'OmniAudio_noBG_Paralleltask':
        args.dataset_cls = OmniAudio_noBG_Paralleltask
        args.train_batch_size = 4#args.bs_mult * args.ngpu
        args.val_batch_size = 4
    elif args.dataset == 'OmniAudio_noBG_Paralleltask_depth':
        args.dataset_cls = OmniAudio_noBG_Paralleltask_depth
        args.train_batch_size = 4#args.bs_mult * args.ngpu
        args.val_batch_size = 4
    elif args.dataset == 'OmniAudio_noBG_Paralleltask_depth_noSeman':
        args.dataset_cls = OmniAudio_noBG_Paralleltask_depth_noSeman
        args.train_batch_size = 4#args.bs_mult * args.ngpu
        args.val_batch_size = 4
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    # Readjust batch size to mini-batch size for apex
    if args.apex:
        args.train_batch_size = args.bs_mult
        args.val_batch_size = args.bs_mult_val

    
    args.num_workers = 4 * args.ngpu
    if args.test_mode:
        args.num_workers = 1


    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           False,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=args.dataset_cls.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # Image appearance transformations
    train_input_transform = []
    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass



    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()
    
    if args.jointwtborder: 
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(args.dataset_cls.ignore_label, 
            args.dataset_cls.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'OmniAudio_noBG_Paralleltask':
        eval_size = 960
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        train_set = args.dataset_cls.OmniAudio(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform)
        val_set = args.dataset_cls.OmniAudio(
            'semantic', 'val',
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform)

    elif args.dataset == 'OmniAudio_noBG_Paralleltask_depth':
        eval_size = 960
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        train_set = args.dataset_cls.OmniAudio(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform)
        val_set = args.dataset_cls.OmniAudio(
            'semantic', 'val',
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform)


    elif args.dataset == 'OmniAudio_noBG_Paralleltask_depth_noSeman':
        eval_size = 960
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        train_set = args.dataset_cls.OmniAudio(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform)
        val_set = args.dataset_cls.OmniAudio(
            'semantic', 'val',
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform)
        
    elif args.dataset == 'null_loader':
        train_set = args.dataset_cls.null_loader(args.crop_size)
        val_set = args.dataset_cls.null_loader(args.crop_size)
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))
    
    if args.apex:
        from datasets.sampler import DistributedSampler
        train_sampler = DistributedSampler(train_set, pad=True, permutation=True, consecutive_sample=False)
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False, sampler = val_sampler)

    return train_loader, val_loader,  train_set

