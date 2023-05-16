# Modified from: https://github.com/facebookresearch/r3m/blob/main/r3m/models/models_r3m.py on 2023-03-19
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
from torchvision import transforms
from r3m import utils
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T
import FiLM_resnet

epsilon = 1e-8


def do_nothing(x):
    return x


class R3M(nn.Module):
    def __init__(
        self,
        device,
        lr,
        hidden_dim,
        size=34,
        l2weight=1.0,
        l1weight=1.0,
        langweight=1.0,
        tcnweight=0.0,
        l2dist=True,
        bs=16,
    ):
        super().__init__()

        self.device = device
        self.use_tb = False
        self.l2weight = l2weight
        self.l1weight = l1weight
        self.tcnweight = tcnweight  # Weight on TCN loss (states closer in same clip closer in embedding)
        self.l2dist = l2dist  # Use -l2 or cosine sim
        self.langweight = langweight  # Weight on language reward
        self.size = size  # Size ResNet or ViT
        self.num_negatives = 3

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
            self.convnet = AutoModel.from_config(
                config=AutoConfig.from_pretrained("google/vit-base-patch32-224-in21k")
            ).to(self.device)

        if self.size == 0:
            self.normlayer = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        else:
            self.normlayer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())

        # Language Reward
        if self.langweight > 0.0:
            # Pretrained DistilBERT Sentence Encoder
            from r3m.models.models_language import LangEncoder, LanguageReward

            self.lang_enc = LangEncoder(self.device, 0, 0)
            self.lang_rew = LanguageReward(
                None, self.outdim, hidden_dim, self.lang_enc.lang_size, simfunc=self.sim
            )
            params += list(self.lang_rew.parameters())
        ########################################################################

        # Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    def get_reward(self, e0, es, sentences):
        # Only callable is langweight was set to be 1
        le = self.lang_enc(sentences)
        return self.lang_rew(e0, es, le)

    # Forward Call (im --> representation)
    def forward(self, obs, num_ims=1, obs_shape=[3, 224, 224]):
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

        # Input must be [0, 255], [3,244,244]
        obs = obs.float() / 255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        else:
            d = self.cs(tensor1, tensor2)
        return d


device = "cuda" if torch.cuda.is_available() else "cpu"


def cleanup_config(cfg):
    VALID_ARGS = [
        "_target_",
        "device",
        "lr",
        "hidden_dim",
        "size",
        "l2weight",
        "l1weight",
        "langweight",
        "tcnweight",
        "l2dist",
        "bs",
    ]

    import copy

    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    # config.agent["_target_"] = "r3m.R3M"
    config.agent["_target_"] = "FiLM_R3M.R3M"
    config["device"] = device

    # Hardcodes to remove the language head
    # Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent


def remove_language_head(state_dict):
    keys = state_dict.keys()
    # Hardcodes to remove the language head
    # Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict


def load_r3m(modelid):
    import os
    from os.path import expanduser
    import gdown
    import omegaconf
    import hydra

    # home = os.path.join(expanduser("~"), ".r3m")
    home = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".r3m")
    if modelid == "resnet50":
        foldername = "r3m_50"
        modelurl = "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA"
        configurl = "https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8"
    elif modelid == "resnet34":
        foldername = "r3m_34"
        modelurl = "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE"
        configurl = "https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW"
    elif modelid == "resnet18":
        foldername = "r3m_18"
        modelurl = "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-"
        configurl = "https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6"
    else:
        raise NameError("Invalid Model ID")

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(
        torch.load(modelpath, map_location=torch.device(device))["r3m"]
    )
    rep.load_state_dict(r3m_state_dict)
    return rep


if __name__ == "__main__":
    from torchinfo import summary
    import r3m

    # Create model
    model = load_r3m("resnet50").to(device)
    summary(model, input_size=(1, 3, 224, 224), depth=float("inf"), device=device)

    # Create reference model
    reference_model = r3m.load_r3m("resnet50").to(device)

    # Compare outputs
    example_input = torch.rand(1, 3, 224, 224, device=device)
    example_output = model(example_input * 255.0)
    reference_output = reference_model(example_input * 255.0)
    print(
        f"Output difference with reference: {torch.norm(example_output - reference_output)}"
    )

    # Compare outputs with convnet
    convnet = model.module.convnet
    normlayer = model.module.normlayer
    num_params = sum([sum(x) for x in convnet.num_planes_per_block_per_layer])
    example_output_2 = convnet(normlayer(example_input))
    example_output_3 = convnet(
        normlayer(example_input),
        beta=torch.zeros(1, num_params, device=device),
        gamma=torch.ones(1, num_params, device=device),
    )

    print(
        f"Output difference with convnet: {torch.norm(example_output - example_output_2)}"
    )
    print(
        f"Output difference with convnet beta gamma: {torch.norm(example_output - example_output_3)}"
    )
