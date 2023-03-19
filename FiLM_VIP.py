# Modified from: https://github.com/facebookresearch/vip/blob/main/vip/models/model_vip.py on 2023-03-19
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import copy
from torch.hub import load_state_dict_from_url
import gdown
import hydra
import omegaconf
from os.path import expanduser
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T
import FiLM_resnet


class VIP(nn.Module):
    def __init__(self, device="cuda", lr=1e-4, hidden_dim=1024, size=50, l2weight=1.0, l1weight=1.0, gamma=0.98, num_negatives=0):
        super().__init__()
        self.device = device
        self.l2weight = l2weight
        self.l1weight = l1weight

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.size = size  # Resnet size
        self.num_negatives = num_negatives

        # Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        # Sub Modules
        # Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = FiLM_resnet.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = FiLM_resnet.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = FiLM_resnet.resnet50(pretrained=False)
        elif size == 0:
            from transformers import AutoConfig
            self.outdim = 768
            self.convnet = AutoModel.from_config(config=AutoConfig.from_pretrained(
                'google/vit-base-patch32-224-in21k')).to(self.device)

        if self.size == 0:
            self.normlayer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if hidden_dim > 0:
            self.convnet.fc = nn.Linear(self.outdim, hidden_dim)
        else:
            self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())

        # Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    # Forward Call (im --> representation)
    def forward(self, obs, obs_shape=[3, 224, 224]):
        # obs_shape = obs.shape[1:]
        # if not already resized and cropped, then add those in preprocessing
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )
        # Input must be [0, 255], [3,224,224]
        obs = obs.float() / 255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        return d


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "num_negatives"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "FiLM_VIP.VIP"
    config["device"] = device

    return config.agent


def load_vip(modelid='resnet50'):
    # home = os.path.join(expanduser("~"), ".vip")
    home = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vip")

    if not os.path.exists(os.path.join(home, modelid)):
        os.makedirs(os.path.join(home, modelid))
    folderpath = os.path.join(home, modelid)
    modelpath = os.path.join(home, modelid, "model.pt")
    configpath = os.path.join(home, modelid, "config.yaml")

    # Default download from PyTorch S3 bucket; use G-Drive as a backup.
    try:
        if modelid == "resnet50":
            modelurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
            configurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/config.yaml"
        else:
            raise NameError('Invalid Model ID')
        if not os.path.exists(modelpath):
            load_state_dict_from_url(modelurl, folderpath)
            load_state_dict_from_url(configurl, folderpath)
    except:
        if modelid == "resnet50":
            modelurl = 'https://drive.google.com/uc?id=1LuCFIV44xTZ0GLmLwk36BRsr9KjCW_yj'
            configurl = 'https://drive.google.com/uc?id=1XSQE0gYm-djgueo8vwcNgAiYjwS43EG-'
        else:
            raise NameError('Invalid Model ID')
        if not os.path.exists(modelpath):
            gdown.download(modelurl, modelpath, quiet=False)
            gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    vip_state_dict = torch.load(modelpath, map_location=torch.device(device))['vip']
    rep.load_state_dict(vip_state_dict)
    return rep


if __name__ == "__main__":
    from torchinfo import summary

    # Load model
    model = load_vip("resnet50")
    summary(model, input_size=(1, 3, 224, 224), depth=float('inf'), device=device)

    # Run model
    example_input = torch.rand(1, 3, 224, 224, device=device)
    example_output = model(example_input * 255.)

    # Run model with convnet and normlayer
    convnet = model.module.convnet
    normlayer = model.module.normlayer

    num_params = sum([sum(x) for x in convnet.num_planes_per_block_per_layer])
    example_output_2 = convnet(normlayer(example_input))
    print(f"Output difference with convnet: {torch.norm(example_output - example_output_2)}")

    # Run model with defined beta and gamma
    example_output_3 = convnet(normlayer(example_input), beta=torch.zeros(1, num_params, device=device),
                               gamma=torch.ones(1, num_params, device=device))
    print(f"Output difference with convnet beta gamma: {torch.norm(example_output - example_output_3)}")
