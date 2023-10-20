import copy
import glob
import re
from collections import defaultdict

import numpy as np
import torch
class Clients:
    def __init__(self,path):
        self.path = path
        self.local_updated_weights = {}
        self.user_ids = []
        self.patternName = r".*_\D*(\d+).npz"
    def load_updates_from_files(self,epoch):
        if epoch == -1:
            local_img_paths = glob.glob(f'{self.path}/*[0-9]*.npz')
        else:
            local_img_paths = glob.glob(f'{self.path}/epoch_{epoch}/*[0-9]*.npz')
        local_updated_weights = {}
        user_ids = []
        for client_file_path in local_img_paths:
            client_id = int(re.findall(self.patternName, client_file_path)[0])
            user_ids.append(client_id)
            # client = np.load(f'{client_file_path}', allow_pickle=True)['arr_0'].tolist()
            client = np.load(f'{client_file_path}', allow_pickle=True)['local_update_weight'].tolist()
            local_updated_weights[client_id] = client
        # global_updates = np.mean(list(local_updated_weights.values()), axis=0)
        return local_updated_weights,user_ids
    def reset_local_updated_weights(self):
        self.local_updated_weights = {}
        self.user_ids=[]
        self.local_models = []

    def set_local_updated_weight(self,user,local_updated_weight):
        # self.local_updated_weights[user] = copy.deepcopy(local_updated_weight)
        self.local_updated_weights[user] = defaultdict()
        for key,value in local_updated_weight.items():
            if torch.is_tensor(value):
                self.local_updated_weights[user][copy.deepcopy(key)] = copy.deepcopy(value).cpu().numpy()
            else:
                self.local_updated_weights[user][copy.deepcopy(key)] = copy.deepcopy(value)
        self.user_ids.append(user)
