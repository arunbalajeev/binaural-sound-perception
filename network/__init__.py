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
Network Initializations
"""

import logging
import importlib
import torch



def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    #for param in net.parameters():
    #    print(param)
    #    print('{:2.1f}M'.format(param.nelement()/1000000))
    #num_params1 = sum([param.nelement() for param in net.parameters() if param=])
    #num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def warp_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    print(num_classes)
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)
    return net
