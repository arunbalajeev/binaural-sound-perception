"""
Dataset setup and loaders
"""
from datasets import cityscapes
from datasets import mapillary
from datasets import kitti
from datasets import camvid
from datasets import OmniAudio
from datasets import OmniAudio_woSkipConn
from datasets import OmniAudio_audioBG
from datasets import OmniAudio_audioBG_fullSeg
from datasets import OmniAudio_audioBG_fullSeg_Logmag
from datasets import OmniAudio_audioBG_fullSeg_noSkip
from datasets import OmniAudio_audioBG_fullSeg_noaudio
from datasets import eval_OmniAudio_audioBG_fullSeg_Logmag
from datasets import OmniAudio_audioBG_fullSeg_Logmag_diffmask
from datasets import OmniAudio_audioBG_SeqSpec
from datasets import eval_OmniAudio_audioBG_fullSeg_noSkip
from datasets import OmniAudio_audioBG_SeqSpec_diffmask
from datasets import eval_OmniAudio_audioBG_SeqSpec
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_seq2seq
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Comp
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_Comp
from datasets import OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask_Comp
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Sem_Comp
from datasets import OmniAudio_4ch_audioBG_fullSeg_Logmag_diffmask
from datasets import OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_Comp_3class
from datasets import OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp_3class
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class
from datasets import OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class
from datasets import OmniAudio_1ch_audioBG_SeqSpec_diffmask_Comp_3class
from datasets import OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class_aseq
from datasets import OmniAudio_audioBG_Spec_diffmask_seq2seq_Comp
from datasets import OmniAudio_audioBG_Spec_diffmask_seq2seq_Semmask_Comp
from datasets import OmniAudio_4ch_audioBG_Spec_diffmask_Comp_3class
from datasets import OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_tilt
from datasets import OmniAudio_1ch_audioBG_Spec_diffmask_Comp_3class
from datasets import OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_teacher
from datasets import OmniAudio_8ch_audioBG_Spec_diffmask_Comp_3class
from datasets import OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBGteacher_Paralleltask_3class
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class
from datasets import OmniAudio_2ch_audioBG_depth_Logmag_diffmask
from datasets import OmniAudio_audioBG_SeqSpec_depth_diffmask_Comp_Paralleltask_3class
from datasets import OmniAudio_audioBG_ranking_teacher
from datasets import OmniAudio_1ch_audioBG_fullSeg_Logmag_diffmask
from datasets import OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask_tilt
from datasets import OmniAudio_8ch_audioBG_fullSeg_Logmag_diffmask
from datasets import OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_Valranking
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_Quad
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_noSeman
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_6mic
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_6mic
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_4mic
from datasets import OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_4mic
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
    if args.dataset == 'OmniAudio':
        args.dataset_cls = OmniAudio
        args.train_batch_size = 8#args.bs_mult * args.ngpu
        args.val_batch_size = 4
    elif args.dataset == 'OmniAudio_woSkipConn':
        args.dataset_cls = OmniAudio_woSkipConn
        args.train_batch_size = 8#args.bs_mult * args.ngpu
        args.val_batch_size = 4
    elif args.dataset == 'OmniAudio_audioBG':
        args.dataset_cls = OmniAudio_audioBG
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg':
        args.dataset_cls = OmniAudio_audioBG_fullSeg
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_noSkip':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_noSkip
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_noaudio':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_noaudio
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'eval_OmniAudio_audioBG_fullSeg_Logmag':
        args.dataset_cls = eval_OmniAudio_audioBG_fullSeg_Logmag
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'eval_OmniAudio_audioBG_fullSeg_noSkip':
        args.dataset_cls = eval_OmniAudio_audioBG_fullSeg_noSkip
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'eval_OmniAudio_audioBG_SeqSpec':
        args.dataset_cls = eval_OmniAudio_audioBG_SeqSpec
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_seq2seq
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Comp':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask_Comp':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Sem_Comp':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Sem_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_4ch_audioBG_fullSeg_Logmag_diffmask':
        args.dataset_cls = OmniAudio_4ch_audioBG_fullSeg_Logmag_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp':
        args.dataset_cls = OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_1ch_audioBG_SeqSpec_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_1ch_audioBG_SeqSpec_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class_aseq':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class_aseq
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_seq2seq_Comp':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_seq2seq_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_seq2seq_Semmask_Comp':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_seq2seq_Semmask_Comp
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_4ch_audioBG_Spec_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_4ch_audioBG_Spec_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_tilt':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_tilt
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_1ch_audioBG_Spec_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_1ch_audioBG_Spec_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_teacher':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_teacher
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_8ch_audioBG_Spec_diffmask_Comp_3class':
        args.dataset_cls = OmniAudio_8ch_audioBG_Spec_diffmask_Comp_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask':
        args.dataset_cls = OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBGteacher_Paralleltask_3class':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBGteacher_Paralleltask_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_noBG_Paralleltask':
        args.dataset_cls = OmniAudio_noBG_Paralleltask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_2ch_audioBG_depth_Logmag_diffmask':
        args.dataset_cls = OmniAudio_2ch_audioBG_depth_Logmag_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_depth_diffmask_Comp_Paralleltask_3class':
        args.dataset_cls = OmniAudio_audioBG_SeqSpec_depth_diffmask_Comp_Paralleltask_3class
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_ranking_teacher':
        args.dataset_cls = OmniAudio_audioBG_ranking_teacher
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_1ch_audioBG_fullSeg_Logmag_diffmask':
        args.dataset_cls = OmniAudio_1ch_audioBG_fullSeg_Logmag_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask_tilt':
        args.dataset_cls = OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask_tilt
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_8ch_audioBG_fullSeg_Logmag_diffmask':
        args.dataset_cls = OmniAudio_8ch_audioBG_fullSeg_Logmag_diffmask
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_Valranking':
        args.dataset_cls = OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_Valranking
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_noBG_Paralleltask_depth':
        args.dataset_cls = OmniAudio_noBG_Paralleltask_depth
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_Quad':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_Quad
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_noSeman':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_noSeman
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_noBG_Paralleltask_depth_noSeman':
        args.dataset_cls = OmniAudio_noBG_Paralleltask_depth_noSeman
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_6mic':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_6mic
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_6mic':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_6mic
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_4mic':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_4mic
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_4mic':
        args.dataset_cls = OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_4mic
        args.train_batch_size = 2#args.bs_mult * args.ngpu
        args.val_batch_size = 2
    elif args.dataset == 'cityscapes':
        args.dataset_cls = cityscapes
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    elif args.dataset == 'mapillary':
        args.dataset_cls = mapillary
        args.train_batch_size = args.bs_mult * args.ngpu
        args.val_batch_size = 4
    elif args.dataset == 'ade20k':
        args.dataset_cls = ade20k
        args.train_batch_size = args.bs_mult * args.ngpu
        args.val_batch_size = 4
    elif args.dataset == 'kitti':
        args.dataset_cls = kitti
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    elif args.dataset == 'camvid':
        args.dataset_cls = camvid
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    elif args.dataset == 'null_loader':
        args.dataset_cls = null_loader
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
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

    if args.dataset == 'OmniAudio':
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

    elif args.dataset == 'OmniAudio_woSkipConn':
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


    elif args.dataset == 'OmniAudio_audioBG':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_noSkip':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_noaudio':
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

    elif args.dataset == 'eval_OmniAudio_audioBG_fullSeg_Logmag':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec':
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

    elif args.dataset == 'eval_OmniAudio_audioBG_fullSeg_noSkip':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask':
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

    elif args.dataset == 'eval_OmniAudio_audioBG_SeqSpec':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Comp':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Semmask_Comp':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_seq2seq_Sem_Comp':
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

    elif args.dataset == 'OmniAudio_4ch_audioBG_fullSeg_Logmag_diffmask':
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

    elif args.dataset == 'OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_4ch_audioBG_SeqSpec_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_1ch_audioBG_SeqSpec_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_diffmask_Comp_Paralleltask_3class_aseq':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_seq2seq_Comp':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_seq2seq_Semmask_Comp':
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

    elif args.dataset == 'OmniAudio_4ch_audioBG_Spec_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_tilt':
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

    elif args.dataset == 'OmniAudio_1ch_audioBG_Spec_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_teacher':
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

    elif args.dataset == 'OmniAudio_8ch_audioBG_Spec_diffmask_Comp_3class':
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

    elif args.dataset == 'OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBGteacher_Paralleltask_3class':
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

    elif args.dataset == 'OmniAudio_noBG_Paralleltask':
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

    elif args.dataset == 'OmniAudio_2ch_audioBG_depth_Logmag_diffmask':
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

    elif args.dataset == 'OmniAudio_audioBG_SeqSpec_depth_diffmask_Comp_Paralleltask_3class':
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

    elif args.dataset == 'OmniAudio_audioBG_ranking_teacher':
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

    elif args.dataset == 'OmniAudio_1ch_audioBG_fullSeg_Logmag_diffmask':
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

    elif args.dataset == 'OmniAudio_2ch_audioBG_fullSeg_Logmag_diffmask_tilt':
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

    elif args.dataset == 'OmniAudio_8ch_audioBG_fullSeg_Logmag_diffmask':
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

    elif args.dataset == 'OmniAudio_audioBG_fullSeg_Logmag_diffmask_Comp_3class_Valranking':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_Quad':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_6mic':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_6mic':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_4mic':
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

    elif args.dataset == 'OmniAudio_audioBG_Spec_diffmask_Comp_noBG_Paralleltask_3class_depth_4mic':
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

    elif args.dataset == 'cityscapes':
        city_mode = 'train' ## Can be trainval
        city_quality = 'fine'
        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None
            train_set = args.dataset_cls.CityScapesUniform(
                city_quality, city_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes)
        else:
            train_set = args.dataset_cls.CityScapes(
                city_quality, city_mode, 0, 
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv)

        val_set = args.dataset_cls.CityScapes('fine', 'val', 0, 
                                              transform=val_input_transform,
                                              target_transform=target_transform,
                                              cv_split=args.cv)
    elif args.dataset == 'mapillary':
        eval_size = 1536
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        train_set = args.dataset_cls.Mapillary(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode)
        val_set = args.dataset_cls.Mapillary(
            'semantic', 'val',
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False)
    elif args.dataset == 'ade20k':
        eval_size = 384
        val_joint_transform_list = [
                joint_transforms.ResizeHeight(eval_size),
  		joint_transforms.CenterCropPad(eval_size)]
            
        train_set = args.dataset_cls.ade20k(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode)
        val_set = args.dataset_cls.ade20k(
            'semantic', 'val',
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False)
    elif args.dataset == 'kitti':
        # eval_size_h = 384
        # eval_size_w = 1280
        # val_joint_transform_list = [
        #         joint_transforms.ResizeHW(eval_size_h, eval_size_w)]
            
        train_set = args.dataset_cls.KITTI(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm)
        val_set = args.dataset_cls.KITTI(
            'semantic', 'trainval', 0, 
            joint_transform_list=None,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    elif args.dataset == 'camvid':
        # eval_size_h = 384
        # eval_size_w = 1280
        # val_joint_transform_list = [
        #         joint_transforms.ResizeHW(eval_size_h, eval_size_w)]
            
        train_set = args.dataset_cls.CAMVID(
            'semantic', 'trainval', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm)
        val_set = args.dataset_cls.CAMVID(
            'semantic', 'test', 0, 
            joint_transform_list=None,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)

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

