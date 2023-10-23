import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict
import torch
@dataclass
class Params:
    # general arguments
    path : str = None
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # arguments for fl
    model : str = None
    noniid  : bool = False
    clients_num : int = 100

    # arguments for local anomaly detection
    lambda_signv: float = 3.0
    lambda_sortv: float = 3.0
    lambda_classv: float = 3.0

    # arguments for task detection
    tau: float = -0.9
    corr_detecting: bool = False
    corr_first_num: int = 1

    # arguments for domain detection
    save_reference: bool = True
    reference_path: str = None
    reference_round: int = 3
    pretrain_epoch: int = 100
    epoch: int = 5
    update_epoch: int = 0
    weight_decay : float = 1e-6
    lr : float = 5e-8
    pretrain_lr : float = 2e-3
    save_model_path : str = "result/model"

    def check(self):
        if self.path is None or not os.path.exists(self.path):
            print(f"The detection path can not be found, you should check whether the path is correct.")
            exit()

        if self.model is None or self.model not in ["resnet", "vgg", "vit"]:
            print(f"Please choose the model type in resnet, vgg, or vit.")
            exit()


