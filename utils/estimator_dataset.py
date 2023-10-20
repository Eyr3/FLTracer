import os
import re
import glob
import copy
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from utils.utils import *
from utils.CAM import ConvolutionAnomalyMatrix
from utils.Client import Clients
class ClientsDataset():
    def __init__(self, args,measurement=False):
        self.measurement = measurement
        self.noniid = args.noniid
        self.path = args.reference_path
        self.client_pre_imgs = defaultdict()
        self.global_pre_imgs = defaultdict()
        self.reference_round = args.reference_round
        self.clients_num = args.clients_num
        self.sort_sim = 'footrule_mult'
        self.model_name = args.model
        self.layers = None
        representations, self.reference_global_cam=self.calculate_ddist()

        self.velocity = representations['global_updates']#[int(len(representations['global_updates'])/2):]
        # self.velocity = self.velocity.reshape(self.velocity.shape[0],self.clients_num,self.reference_round)
        self.position = representations['local_updates']#[int(len(representations['global_updates'])/2):]
        # self.position = self.position.reshape(self.position.shape[0], self.clients_num, self.reference_round)
        self.k7 = representations['other_clients_updates']#[int(len(representations['global_updates'])/2):]
        self.layer_num = len(self.position)
        # self.x = np.stack([self.position,self.velocity])
        self.x = np.concatenate([self.position,self.velocity])
    def get_client_x(self,client_id):
        client_x=np.concatenate([self.position[:,client_id,:],self.velocity[:,client_id,:]])
        return  client_x
    def calculate_ddist(self):
        epoch_list_name = os.listdir(self.path)
        epoch_list = []
        for s in epoch_list_name:
            match = re.search(r'\d+', s)
            if match:
                epoch_list.append(int(match.group()))

        epoch_list.sort()
        start_epoch = min(epoch_list)
        local_updated_weights, user_ids = Clients(self.path).load_updates_from_files(start_epoch)
        CAM = ConvolutionAnomalyMatrix(local_updated_weights, user_ids, noniid=self.noniid,measurement=self.measurement)
        pre_global_cam = CAM.global_cam
        self.layers = CAM.layers

        # layers = layers[:26]
        self.client_mean_imgs = defaultdict(list)
        self.global_mean_imgs = defaultdict(list)

        for layer in self.layers:
            self.client_pre_imgs[layer] = [[] for i in range(self.clients_num)]

        self.other_client_id_and_epoch = [{"epoch": [], "client_ids": []} for i in range(self.clients_num)]
        self.global_pre_imgs = copy.deepcopy(self.client_pre_imgs)


        for epoch in epoch_list[1:]:
            local_updated_weights, user_ids = Clients(self.path).load_updates_from_files(epoch)
            CAM = ConvolutionAnomalyMatrix(local_updated_weights, user_ids, noniid=self.noniid,measurement=self.measurement)
            clients_cam = CAM.clients_cam
            for layer_index,layer in enumerate(self.layers):
                layer_cams = clients_cam[layer_index].tolist()
                layer_cams.append(pre_global_cam[layer])
                result_imgs = np.argsort(np.argsort(layer_cams, axis=0), axis=0)

                for i, client in enumerate(user_ids):
                    self.client_pre_imgs[layer][client].append(result_imgs[i])
                    self.global_pre_imgs[layer][client].append(result_imgs[-1])
                    if epoch not in self.other_client_id_and_epoch[client]['epoch']:
                        self.other_client_id_and_epoch[client]['epoch'].append(epoch)
                        self.other_client_id_and_epoch[client]['client_ids'].append(list(set(user_ids) - set([client])))
                    # self.client_pre_imgs['epoch'][client].append(epoch)
                    if len(self.client_pre_imgs[layer][client]) > self.reference_round:
                        del self.client_pre_imgs[layer][client][0]
                        del self.global_pre_imgs[layer][client][0]
            pre_global_cam = CAM.global_cam


        for indx, layer in enumerate(CAM.layers):
            self.client_mean_imgs[layer] = np.mean(np.array(self.client_pre_imgs[layer]), axis=1)
            self.global_mean_imgs[layer] = np.mean(np.array(self.global_pre_imgs[layer]), axis=1)
        representations = self.get_upper_limits()
        return representations,pre_global_cam

    def get_upper_limits(self):
        local_updates_represenations = []
        global_updates_represenations = []
        other_clients_imgs=[]
        for layer in self.layers:
            other_clients_imgs_mean=[[] for i in range(self.clients_num)]
            shape0, shape1 = self.client_pre_imgs[layer][0][0].shape
            results = get_sim(self.sort_sim, np.array(self.client_pre_imgs[layer]).reshape(
                self.reference_round * self.clients_num, shape0, shape1),
                              np.repeat(np.expand_dims(self.client_mean_imgs[layer], 1), self.reference_round, axis=1).reshape(
                                  self.reference_round * self.clients_num, shape0, shape1)).reshape(self.clients_num,self.reference_round)
            local_updates_represenations.append(results)

            tmp_all_clients = np.zeros((self.clients_num,),dtype=int)
            for client_id in range(self.clients_num):
                for count in range(self.reference_round):
                    other_client_ids = np.array(self.other_client_id_and_epoch[client_id]['client_ids'][count])
                    other_clients_imgs_mean[client_id].append(np.mean(results[other_client_ids,np.array(tmp_all_clients[other_client_ids]/9,dtype=int)]))
                    tmp_all_clients[other_client_ids] += 1

            other_clients_imgs.append(other_clients_imgs_mean)


            results = get_sim(self.sort_sim, np.array(self.global_pre_imgs[layer]).reshape(
                self.reference_round * self.clients_num, shape0, shape1),
                              np.repeat(np.expand_dims(self.global_mean_imgs[layer], 1), self.reference_round,
                                        axis=1).reshape(
                                  self.reference_round * self.clients_num, shape0, shape1)).reshape(self.clients_num,self.reference_round)
            global_updates_represenations.append(results)
        return {'local_updates': np.array(local_updates_represenations),
                'global_updates': np.array(global_updates_represenations),
                'other_clients_updates': np.array(other_clients_imgs)}

class ClientDataLoader(Dataset):
    def __init__(self,z,x_head,u_t_0,client_id=-1):
        super().__init__()
        self.x=x_head
        self.z=z #measurment
        self.u_t_0=u_t_0
        self.client_id=client_id
    def __getitem__(self, idx):
        if self.client_id != -1:
            x_t = torch.Tensor(self.x[:, self.client_id, idx + 1])
            x_t_1 = torch.Tensor(self.x[:, self.client_id, idx])
            z_t = torch.Tensor(self.z[:, self.client_id, idx + 1])
            u_t_0 = torch.Tensor(self.u_t_0[:, self.client_id, idx + 1])
            return x_t_1.unsqueeze(0), x_t.unsqueeze(0),z_t.unsqueeze(0),u_t_0.unsqueeze(0)
        else:
            x_t = torch.Tensor(self.x[:, :, idx+1])
            x_t_1 = torch.Tensor(self.x[:, :, idx])
            z_t = torch.Tensor(self.z[:, :, idx+1])
            u_t_0 = torch.Tensor(self.u_t_0[:, :, idx + 1])
            return x_t_1.T,x_t.T,z_t.T, u_t_0.T
    def __len__(self):
        return self.x.shape[-1]-1
class ClientDataLoader_update(Dataset):
    def __init__(self,x_head,u_t_0):
        super().__init__()
        self.x=x_head
        self.u_t_0=u_t_0
    def __getitem__(self, idx):
        x_t = torch.Tensor(self.x[:,  idx + 1])
        x_t_1 = torch.Tensor(self.x[:,  idx])
        u_t_0 = torch.Tensor(self.u_t_0[:, idx + 1])
        return x_t_1.unsqueeze(0), x_t.unsqueeze(0),u_t_0.unsqueeze(0)

    def __len__(self):
        return self.x.shape[-1]-1
