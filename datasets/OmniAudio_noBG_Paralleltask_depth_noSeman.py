"""
Cityscapes Dataset Loader
"""
import logging
import json,torch
import os, glob, librosa
import numpy as np
from PIL import Image
from torch.utils import data

import torchvision.transforms as transforms
import datasets.uniform as uniform
import datasets.cityscapes_labels as cityscapes_labels

from config import cfg

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
num_classes = 19
ignore_label = 255#[255,1,2,3,4,5,6,7,8,9,10,12]
root = cfg.DATASET.OMNIAUDIO_DIR
aug_root = cfg.DATASET.CITYSCAPES_AUG_DIR

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def add_items(items, aug_items, cities, img_path, mask_path, mask_postfix, mode, maxSkip):
    """

    Add More items ot the list from the augmented dataset
    """

    for c in cities:
        c_items = [name.split('_leftImg8bit.png')[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
                    os.path.join(mask_path, c, it + mask_postfix))
            ########################################################
            ###### dataset augmentation ############################
            ########################################################
            if mode == "train" and maxSkip > 0:
                new_img_path = os.path.join(aug_root, 'leftImg8bit_trainvaltest', 'leftImg8bit')
                new_mask_path = os.path.join(aug_root, 'gtFine_trainvaltest', 'gtFine')
                file_info = it.split("_")
                cur_seq_id = file_info[-1]

                prev_seq_id = "%06d" % (int(cur_seq_id) - maxSkip)
                next_seq_id = "%06d" % (int(cur_seq_id) + maxSkip)
                prev_it = file_info[0] + "_" + file_info[1] + "_" + prev_seq_id
                next_it = file_info[0] + "_" + file_info[1] + "_" + next_seq_id
                prev_item = (os.path.join(new_img_path, c, prev_it + '_leftImg8bit.png'),
                             os.path.join(new_mask_path, c, prev_it + mask_postfix))
                if os.path.isfile(prev_item[0]) and os.path.isfile(prev_item[1]):
                    aug_items.append(prev_item)
                next_item = (os.path.join(new_img_path, c, next_it + '_leftImg8bit.png'),
                             os.path.join(new_mask_path, c, next_it + mask_postfix))
                if os.path.isfile(next_item[0]) and os.path.isfile(next_item[1]):
                    aug_items.append(next_item)
            items.append(item)
    # items.extend(extra_items)


def make_cv_splits(img_dir_name):
    """
    Create splits of train/val data.
    A split is a lists of cities.
    split0 is aligned with the default Cityscapes train/val.
    """
    trn_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'train')
    val_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'val')

    trn_cities = ['train/' + c for c in os.listdir(trn_path)]
    val_cities = ['val/' + c for c in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_cities = sorted(trn_cities)

    all_cities = val_cities + trn_cities
    num_val_cities = len(val_cities)
    num_cities = len(all_cities)

    cv_splits = []
    for split_idx in range(cfg.DATASET.CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_cities // cfg.DATASET.CV_SPLITS
        for j in range(num_cities):
            if j >= offset and j < (offset + num_val_cities):
                split['val'].append(all_cities[j])
            else:
                split['train'].append(all_cities[j])
        cv_splits.append(split)

    return cv_splits


def make_split_coarse(img_path):
    """
    Create a train/val split for coarse
    return: city split in train
    """
    all_cities = os.listdir(img_path)
    all_cities = sorted(all_cities)  # needs to always be the same
    val_cities = []  # Can manually set cities to not be included into train split

    split = {}
    split['val'] = val_cities
    split['train'] = [c for c in all_cities if c not in val_cities]
    return split


def make_test_split(img_dir_name):
    test_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'test')
    test_cities = ['test/' + c for c in os.listdir(test_path)]

    return test_cities


def make_dataset(quality, mode,  fine_coarse_mult=6, cv_split=0):
    """
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    audioDict=np.load('/'.join(root.split('/')[:-2])+"/SoundEnergy_60scenes.npy");#print(audioDict)
    assert (quality == 'semantic' and mode in ['train', 'val'])
    if mode == 'train':
        for sc in range(1,115):
            if sc<37:    img_dir_name = 'scene'+str(sc)
            else:    img_dir_name = 'scene%04d'%sc
            check_audioImg_path = os.path.join(root, img_dir_name, 'spectrograms/Track1/')
            audioImg_path = os.path.join(root, img_dir_name, 'spectrograms/Track3/')
            audioImg_path6 = os.path.join(root, img_dir_name, 'spectrograms/Track8/')
            wavaudioImg_path = os.path.join(root, img_dir_name, 'wavsplits/Track3/')
            wavaudioImg_path6 = os.path.join(root, img_dir_name, 'wavsplits/Track8/')
            wavaudioImg_path2 = os.path.join(root, img_dir_name, 'wavsplits/Track7/')
            wavaudioImg_path5 = os.path.join(root, img_dir_name, 'wavsplits/Track4/')
            mask_path = os.path.join(root, img_dir_name,'segment_splitframes/pred/')
            depthmask_path = os.path.join(root, img_dir_name,'monodepth2_front/')
            Sem_mask_path = os.path.join(root, img_dir_name,'binary_semantic_change_masks/')
            img_path = os.path.join(root, img_dir_name,'bg.png')
            mask_postfix = '_mask.png';audio_postfix='.npy';depth_postfix='_disp.npy';
            for it_full in glob.glob(check_audioImg_path+"*.npy"):
                #print(it_full)
                it = it_full.split('/')[-1].split('.')[0]
                if it_full in audioDict.item()[sc]:
                    item = (img_path, os.path.join(mask_path, it+mask_postfix), os.path.join(depthmask_path, it+depth_postfix), os.path.join(Sem_mask_path, it+mask_postfix), os.path.join(audioImg_path, it+audio_postfix), os.path.join(audioImg_path6, it+audio_postfix), wavaudioImg_path+it+'.npy', wavaudioImg_path6+it+'.npy', wavaudioImg_path2+it+'.npy', wavaudioImg_path5+it+'.npy')
                    items.append(item)

    if mode == 'val':
        for sc in range(115,140):
            if sc<37:    img_dir_name = 'scene'+str(sc)
            else:    img_dir_name = 'scene%04d'%sc
            check_audioImg_path = os.path.join(root, img_dir_name, 'spectrograms/Track1/')
            audioImg_path = os.path.join(root, img_dir_name, 'spectrograms/Track3/')
            audioImg_path6 = os.path.join(root, img_dir_name, 'spectrograms/Track8/')
            wavaudioImg_path = os.path.join(root, img_dir_name, 'wavsplits/Track3/')
            wavaudioImg_path6 = os.path.join(root, img_dir_name, 'wavsplits/Track8/')
            wavaudioImg_path2 = os.path.join(root, img_dir_name, 'wavsplits/Track7/')
            wavaudioImg_path5 = os.path.join(root, img_dir_name, 'wavsplits/Track4/')
            mask_path = os.path.join(root, img_dir_name,'segment_splitframes/pred/')
            depthmask_path = os.path.join(root, img_dir_name,'monodepth2_front/')
            Sem_mask_path = os.path.join(root, img_dir_name,'binary_semantic_change_masks/')
            img_path = os.path.join(root, img_dir_name,'bg.png')
            mask_postfix = '_mask.png';audio_postfix='.npy';depth_postfix='_disp.npy';
            for it_full in glob.glob(check_audioImg_path+"*.npy"):
                it = it_full.split('/')[-1].split('.')[0]
                if it_full in audioDict.item()[sc]:
                    item = (img_path, os.path.join(mask_path, it+mask_postfix), os.path.join(depthmask_path, it+depth_postfix), os.path.join(Sem_mask_path, it+mask_postfix), os.path.join(audioImg_path, it+audio_postfix), os.path.join(audioImg_path6, it+audio_postfix), wavaudioImg_path+it+'.npy', wavaudioImg_path6+it+'.npy', wavaudioImg_path2+it+'.npy', wavaudioImg_path5+it+'.npy')
                    items.append(item)

    logging.info('360audio-{}: {} images'.format(mode, len(items)))
    return items

def make_dataset_video():
    """
    Create Filename list for the dataset
    """
    img_dir_name = '../dataset/outdoor/'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit/demoVideo')
    items = []
    categories = os.listdir(img_path)
    for c in categories[1:]:
        c_items = [name.split('_leftImg8bit.png')[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = os.path.join(img_path, c, it + '_leftImg8bit.png')
            items.append(item)
    return items

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

class OmniAudio(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 cv_split=None, eval_mode=False,
                 eval_scales=None, eval_flip=False,class_uniform_pct=0,class_uniform_tile=1024):
        self.quality = quality
        self.mode = mode
        #self.maxSkip = maxSkip
        #self.joint_transform = joint_transform
        #self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        #self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        #if eval_scales != None:
        #    self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        #if cv_split:
        #    self.cv_split = cv_split
        #    assert cv_split < cfg.DATASET.CV_SPLITS, \
        #        'expected cv_split {} to be < CV_SPLITS {}'.format(
        #            cv_split, cfg.DATASET.CV_SPLITS)
        #else:
        #    self.cv_split = 0
        self.imgs = make_dataset(quality, mode, cv_split=0)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, mask,audio1, audio6, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool) + 1):
            imgs = []
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w, h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img = img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(tensor_img)
            return_imgs.append(imgs)
        return return_imgs, mask, audio1, audio6

    def build_epoch(self):
        if self.class_uniform_pct != 0:
            self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                    self.centroids,
                                                    num_classes,
                                                    self.class_uniform_pct)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):

        img_path, mask_path,depthmask_path,sem_mask, audio1, audio6, audio_path1, audio_path6, audio_path2, audio_path5 = self.imgs[index]
        img_name = img_path.split('/')[-2]+"_"+os.path.splitext(os.path.basename(audio1))[0]

        #### W/O Skip connection scratch training ####
        img, mask,depthmask, sem_mask, audio1, audio6, audio_channel1, audio_channel6, audio_channel2, audio_channel5 = Image.open(img_path), Image.open(mask_path),np.load(depthmask_path), Image.open(sem_mask), np.load(audio1), np.load(audio6), np.load(audio_path1), np.load(audio_path6), np.load(audio_path2), np.load(audio_path5)
        img.thumbnail([960,1920], Image.ANTIALIAS);
        #print(audio1.shape)
        ### With Skip Connection and Pretrained model of SOP #####
        #img, mask = np.linalg.norm(np.load(img_path),axis=0), Image.open(mask_path)
        ##img1=np.zeros((img.shape[0],img.shape[1]+64));img1[:img.shape[0],:img.shape[1]] = img
        #mask.thumbnail([960,1920], Image.ANTIALIAS);
        #img=np.expand_dims(img, axis=0)
        audio1=audio1[0,:,:]**2 + audio1[1,:,:]**2;audio1[audio1<1e-5]=1e-5;audio1 = np.log(audio1);audio1=np.expand_dims(audio1, axis=0)
        audio6=audio6[0,:,:]**2 + audio6[1,:,:]**2;audio6[audio6<1e-5]=1e-5;audio6 = np.log(audio6);audio6=np.expand_dims(audio6, axis=0)

        #print(img.shape);print("haja")
        #print(np.array(mask).shape)
        mask = np.array(mask);sem_mask = np.array(sem_mask);depthmask = depthmask[::2,::2]#[:,:,0]
        mask=mask[::2,::2];sem_mask=sem_mask[::2,::2]
        #print(mask[50:60,50:60])
        mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            #if v in [0,1,2,3,4,5,6,7,8,9,10,12]:    mask_copy[mask == k] = 0#255
            if v in [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18]:    mask_copy[mask == k] = 0#255
            else:    mask_copy[mask == k] = v
            mask_copy[mask == k] = v
        mask_copy[sem_mask<128] =0;depthmask=depthmask[0];
        #pos=np.where(np.array(mask_copy) >20);#print(pos)
        #print(mask_copy[50:60,50:60])
        #print(np.unique(mask_copy))
        if self.eval_mode:
            return [transforms.ToTensor()(img)], self._eval_get_item(img, mask_copy,depthmask,sem_mask, audio1, audio6,
                                                                     self.eval_scales,
                                                                     self.eval_flip), [Torch.tensor(sem_mask)], [Torch.tensor(audio1)], [Torch.tensor(audio6)], img_name

        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        #if self.joint_transform is not None:
        #    img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            depthmask = self.target_transform(depthmask)
            sem_mask = self.target_transform(sem_mask)

        audio_diff_spec_2 = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_diff_spec_5 = torch.FloatTensor(generate_spectrogram(audio_channel6 - audio_channel5))
        #mono_aural1 = torch.FloatTensor(generate_spectrogram(audio_channel1))
        #mono_aural6 = torch.FloatTensor(generate_spectrogram(audio_channel6))
        #mono_aural2 = torch.FloatTensor(generate_spectrogram(audio_channel2))
        #mono_aural5 = torch.FloatTensor(generate_spectrogram(audio_channel5))

        return img, depthmask,audio_diff_spec_2,audio_diff_spec_5, sem_mask, audio1, audio6, img_name

    def __len__(self):
        return len(self.imgs)


class CityScapesVideo(data.Dataset):

    def __init__(self, transform=None):
        self.imgs = make_dataset_video()
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs)


