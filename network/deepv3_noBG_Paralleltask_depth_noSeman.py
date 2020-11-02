"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import nn
import functools
from network import SEresnext
from network import Resnet
from network.wider_resnet import wider_resnet38_a2
from network.audio_net_Seq_multitask import AudioNet_multitask
from network.mynn import initialize_weights, Norm2d, Upsample


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'seresnext-50':
            resnet = SEresnext.se_resnext50_32x4d()
        elif trunk == 'seresnext-101':
            resnet = SEresnext.se_resnext101_32x4d()
        elif trunk == 'resnet-50':
            resnet = Resnet.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = Resnet.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256,
                                                       output_stride=8)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + self.skip_num, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final)

    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        xp = self.aspp(x4)

        dec0_up = self.bot_aspp(xp)
        if self.skip == 'm1':
            dec0_fine = self.bot_fine(x1)
            dec0_up = Upsample(dec0_up, x1.size()[2:])
        else:
            dec0_fine = self.bot_fine(x2)
            dec0_up = Upsample(dec0_up, x2.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        main_out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion(main_out, gts)

        return main_out

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

class DeepWV3Plus(nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38', criterion=None,ngf=64, input_nc=1, output_nc=2):

        super(DeepWV3Plus, self).__init__()
        self.criterion = criterion
        self.MSEcriterion= torch.nn.MSELoss()
        logging.info("Trunk: %s", trunk)
        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        try:
            checkpoint = torch.load('/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/semantic-segmentation-master/pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
            wide_resnet.load_state_dict(checkpoint['state_dict'])
            del checkpoint
        except:
            print("=====================Could not load ImageNet weights=======================")
            print("Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models.")

        #audio_unet = AudioNet_multitask(ngf=64,input_nc=2)
        #Acheckpoint = torch.load('/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/audio/audioSynthesis/checkpoints/synBi2Bi_16_25/3_audio.pth', map_location='cpu')
        #pretrained_dict = Acheckpoint
        #model_dict = audio_unet.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k!='audionet_upconvlayer1.0.weight' and k!='audionet_upconvlayer5.0.weight' and k!='audionet_upconvlayer5.0.bias' and k!='conv1x1.0.weight' and k!='conv1x1.0.bias' and k!='conv1x1.1.weight' and k!='conv1x1.1.bias' and k!='conv1x1.1.running_mean' and k!='conv1x1.1.running_var'}
        #model_dict.update(pretrained_dict) 
        #audio_unet.load_state_dict(model_dict)

        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(1024, ngf * 8) #1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 8, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 4, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 2, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf  , output_nc, True) #outermost layer use a sigmoid to bound the mask
        self.conv1x1 = create_conv(4096, 2, 1, 0)

        wide_resnet = wide_resnet.module
        #self.unet= audio_unet
        #print(wide_resnet)
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 64,
                                                       output_stride=8)
        self.depthaspp = _AtrousSpatialPyramidPoolingModule(512,64,
                                                       output_stride=8)

        self.bot_aud1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.bot_multiaud = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(320, 256, kernel_size=1, bias=False)
        self.bot_depthaspp = nn.Conv2d(320, 128, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))
        self.final_depth = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            Norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            Norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=False))

        initialize_weights(self.final);initialize_weights(self.bot_aud1);initialize_weights(self.bot_multiaud);

    def forward_conv(self, x):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        #visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1) #flatten visual feature
        #visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        return audio_conv4feature, audio_conv5feature #m(mask_prediction)

    def forward_SASR(self, x1, x2):
        _,audio_conv5feature = self.forward_conv(x1)
        _,audio_conv5feature2 = self.forward_conv(x2)

        #visual_feat = self.conv1x1(visual_feat)
        #visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1) #flatten visual feature
        #visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        #print(visual_feat.shape)
        audioVisual_feature = torch.cat((audio_conv5feature, audio_conv5feature2), dim=1)
        #print(audioVisual_feature.shape, audio_conv5feature.shape)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        m = nn.ZeroPad2d((0,1,0,0))
        #print(m(audio_upconv1feature).shape, audio_conv4feature.shape)
        audio_upconv2feature = self.audionet_upconvlayer2(m(audio_upconv1feature))
        audio_upconv3feature = self.audionet_upconvlayer3(m(audio_upconv2feature))
        audio_upconv4feature = self.audionet_upconvlayer4(audio_upconv3feature)
        mask_prediction = self.audionet_upconvlayer5(audio_upconv4feature) * 2 - 1

        audio_upconv1feature2 = self.audionet_upconvlayer1(audioVisual_feature)
        m = nn.ZeroPad2d((0,1,0,0))
        #print(audio_upconv1feature.shape, audio_upconv1feature2.shape)
        audio_upconv2feature2 = self.audionet_upconvlayer2(m(audio_upconv1feature2))
        audio_upconv3feature2 = self.audionet_upconvlayer3(m(audio_upconv2feature2))
        audio_upconv4feature2 = self.audionet_upconvlayer4(audio_upconv3feature2)
        mask_prediction2 = self.audionet_upconvlayer5(audio_upconv4feature2) * 2 - 1
        m = nn.ZeroPad2d((0,1,0,1))
        #print(mask_prediction.shape)
        return m(mask_prediction),m(mask_prediction2)


    def forward_Seg(self,x):
        #batch_size, timesteps, C, H, W = x.size()
        #c_in = x.view(batch_size * timesteps, C, H, W)
        audio_conv4feature,_ = self.forward_conv(x)
        #audio_feat = audio_conv5feature.view(audio_conv5feature.shape[0], -1, 1, 1)
        #audio_feat = self.conv1x1(audio_feat)
        #r_in = audio_feat.view(batch_size, timesteps, -1)
        #r_out = self.rnn(r_in)
        return audio_conv4feature

    def forward(self, inp_img, audio1, audio6, gts_diff_2=None, gts_diff_5=None,gts_depth=None):
        '''batch_size, timesteps, C, H, W = audio1.size()
        c_in1 = audio1.view(batch_size * timesteps, C, H, W);c_in2 = audio6.view(batch_size * timesteps, C, H, W);
        audio_conv1feature = self.audionet_convlayer1(c_in1);audio_conv1feature2 = self.audionet_convlayer1(c_in2)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature);audio_conv2feature2 = self.audionet_convlayer2(audio_conv1feature2)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature);audio_conv3feature2 = self.audionet_convlayer3(audio_conv2feature2)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature);audio_conv4feature2 = self.audionet_convlayer4(audio_conv3feature2)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature);audio_conv5feature2 = self.audionet_convlayer5(audio_conv4feature2)
        audio_feat = audio_conv5feature.view(audio_conv5feature.shape[0], -1, 1, 1);audio_feat2 = audio_conv5feature2.view(audio_conv5feature2.shape[0], -1, 1, 1);
        audio_feat = self.conv1x1(audio_feat);audio_feat2 = self.conv1x1(audio_feat2)
        r_in = audio_feat.view(batch_size, timesteps, -1);r_in2 = audio_feat2.view(batch_size, timesteps, -1)
        '''
        out_aud1 = self.forward_Seg(audio1);out_aud6 = self.forward_Seg(audio6)

        #print(inp.size())
        #x_size = inp_img.size()
        #out_aud1=self.unet(audio1);out_aud6 = self.unet(audio6);
        #x = self.mod1(inp_img)
        #m2 = self.mod2(self.pool2(x))
        #x = self.mod3(self.pool3(m2))
        #x = self.mod4(x)
        #x = self.mod5(x)
        #x = self.mod6(x)
        #x = self.mod7(x)
        mask_prediction, mask_prediction2 = self.forward_SASR(audio1, audio6);

        #print(mask_prediction2.shape,gts_diff_5.shape)
        loss = self.MSEcriterion(mask_prediction,gts_diff_2)+ self.MSEcriterion(mask_prediction2,gts_diff_5)

        #x = self.aspp(x)
        #dec0_up = self.bot_aspp(x);print(dec0_up.shape)
        dec0_aud1 =  Upsample(out_aud1, [60,120]);dec0_aud1 = self.bot_aud1(dec0_aud1);
        dec0_aud6 =  Upsample(out_aud6, [60,120]);dec0_aud6 = self.bot_aud1(dec0_aud6);
        dec0_aud = [dec0_aud1, dec0_aud6];dec0_aud = torch.cat(dec0_aud,1);dec0_aud = self.bot_multiaud(dec0_aud);
        #dec0_up = [dec0_up,dec0_aud];dec0_up = torch.cat(dec0_up,1);
        #dec0_auds= self.aspp(dec0_aud);
        dec0_audd = self.depthaspp(dec0_aud);
        #dec0_up = self.bot_aspp(dec0_auds);
        dec0_upd = self.bot_depthaspp(dec0_audd);
        #print(dec0_aud.shape, dec0_up.shape);

        #dec0_fine = self.bot_fine(m2)
        #dec0_up = Upsample(dec0_up,[240,480]);
        dec0_upd = Upsample(dec0_upd, [160,512]);
        #dec0 = [dec0_fine, dec0_up]
        #dec0 = torch.cat(dec0, 1)
        #print(dec0.shape, out_aud1.shape, out_aud6.shape)
        #dec1 = self.final(dec0_up);
        dec1d = self.final_depth(dec0_upd)
        #out = Upsample(dec1,[480,960]);
        outd = Upsample(dec1d, [320,1024])


        #print(out.size())
        #out=self.final(out)
        #print(out.size(),x_size)
        #out = Upsample(out, x_size[1:])
        #print(out.size(),gts.size())
        #print(out[0,0,0:10,0],gts[0,0:10,0])
        if self.training:
            if loss <5.0:
                #print(loss,self.criterion(out, gts))
                return loss+self.MSEcriterion(outd,gts_depth)
            else:
                return self.MSEcriterion(outd,gts_depth)
        return outd


def DeepSRNX50V3PlusD_m1(num_classes, criterion):
    """
    SEResnet 50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='seresnext-50', criterion=criterion, variant='D',
                      skip='m1')

def DeepR50V3PlusD_m1(num_classes, criterion):
    """
    Resnet 50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, variant='D', skip='m1')


def DeepSRNX101V3PlusD_m1(num_classes, criterion):
    """
    SeResnext 101 Based Network
    """
    return DeepV3Plus(num_classes, trunk='seresnext-101', criterion=criterion, variant='D',
                      skip='m1')

