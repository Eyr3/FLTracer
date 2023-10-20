import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from utils.estimator import KalmanFilter_Estimator
from utils.estimator_dataset import ClientsDataset,ClientDataLoader,ClientDataLoader_update
from utils.utils import *
import copy
from filterpy.kalman import KalmanFilter
class DomainDetection:
    def __init__(self,args,clients_cam,user_ids):
        self.clients_cam = clients_cam
        self.user_ids = user_ids
        if args.nocuda:
            self.device = "cpu"
        else:
            self.device = 'cuda:1'
        self.model = args.model

        self.weight_decay = args.weight_decay
        self.pretrain_epoch=args.pretrain_epoch
        self.pretrain_lr = args.pretrain_lr
        self.epoch = args.epoch
        self.lr = args.lr

        self.save_path = args.save_model_path

        # # # reference
        reference_dataset_path = f"{args.reference_path}/reference_dataset.npy"
        if os.path.exists(reference_dataset_path):
            dataset_file = np.load(reference_dataset_path, allow_pickle=True).tolist()
            self.dataset_estimate = dataset_file['estimate']
            self.dataset_measurement = dataset_file['measurement']
        else:
            self.dataset_estimate = ClientsDataset(args)
            self.dataset_measurement = ClientsDataset(args,measurement=True)
            if args.save_reference:
                np.save(reference_dataset_path,{"estimate":self.dataset_estimate,"measurement": self.dataset_measurement})

        self.layers = self.dataset_estimate.layers
        if clients_cam.pre_global_cam is None:
            clients_cam.pre_global_cam = self.dataset_estimate.reference_global_cam
        self.pretrain_loader = ClientDataLoader(self.dataset_measurement.x,self.dataset_estimate.x,self.dataset_estimate.k7)
        self.client_dataloader()
        self.criterion = torch.nn.L1Loss()
        self.estimator = KalmanFilter_Estimator(layer_num=self.dataset_measurement.layer_num)
        pretrain_model = self.estimate(self.estimator,pretrain=True)
        self.clients_estimator = dict()
        for client_id in range(args.clients_num):
            self.clients_estimator[client_id] = self.estimate(copy.deepcopy(pretrain_model), pretrain=False, client_id=client_id)

        # # # detection
        self.sort_sim = 'footrule_mult'
        self.client_mean_imgs = self.dataset_estimate.client_mean_imgs
        self.global_mean_imgs = self.dataset_estimate.global_mean_imgs
        self.velocity = self.dataset_estimate.velocity
        self.velocity = self.velocity.reshape(self.velocity.shape[0], args.clients_num, args.reference_round)
        self.position = self.dataset_estimate.position
        self.position = self.position.reshape(self.position.shape[0], args.clients_num, args.reference_round)
        self.x = np.concatenate([self.position, self.velocity])
        self.x = np.transpose(self.x, (1, 0, 2)).tolist()
        self.u_others = self.dataset_estimate.k7
        self.u_others = np.transpose(self.u_others, (1, 0, 2)).tolist()
        self.update_epoch = args.update_epoch
    def resume_model(self,path):
        state_dict=torch.load(path,map_location=self.device)
        self.estimator.load_state_dict(state_dict)
    def client_dataloader(self):
        self.client_loader=[]
        for client_id in range(self.dataset_measurement.clients_num):
            self.client_loader.append(ClientDataLoader(self.dataset_measurement.x,self.dataset_estimate.x,self.dataset_estimate.k7,client_id=client_id))
    def estimate(self,estimator,pretrain=False,client_id=-1):
        estimator = estimator.to(self.device)
        print(f"Starting estimating...")
        if pretrain:
            data_loader=self.pretrain_loader
            epoch = self.pretrain_epoch
            lr = self.pretrain_lr
        else:
            data_loader = self.client_loader[client_id]
            epoch = self.epoch
            lr = self.lr
        optimizer = torch.optim.Adam([
            {'params': estimator.parameters(), 'lr': lr, 'weight_decay': self.weight_decay,
             'betas': (0.9, 0.999)},
        ])
        # self.test(self.net, self.start_epoch-1)
        best_state = None
        lowest_loss = 10000
        epbar = tqdm(total=epoch)
        for epoch in range(epoch):
            estimator.train()
            avg_loss = []

            for i, (x_t_1,x_t,z_t, u_t_0) in enumerate(data_loader):
                x_t = x_t.cuda(self.device)
                x_t_1 = x_t_1.cuda(self.device)
                z_t = z_t.cuda(self.device)
                u_t_0 = u_t_0.cuda(self.device)

                output_x_t,output_z_t = estimator(x_t_1,u_t_0)
                loss = self.criterion(z_t, output_z_t) + self.criterion(x_t, output_x_t)
                if torch.isnan(loss).any():
                    print("loss is NAN, so stop training...")
                    sys.exit()
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                    best_state = copy.deepcopy(estimator.state_dict())

                print(f"Training total loss: {loss.item()}, if pretrained: {pretrain}")


            epbar.update(1)
        # self.save_model(best_state,pretrain=pretrain,client_id=client_id)
        estimator.load_state_dict(best_state)
        return estimator
    def save_model(self,best_state,client_id=-1,pretrain=False):
        print(f"Saving model to {self.save_path}.")
        os.makedirs(self.save_path,exist_ok=True)
        if pretrain:
            model_name=f"{self.save_path}/pretrain_model.pth"
            torch.save(best_state, model_name)
        elif client_id != -1:
            model_name = f"{self.save_path}/client{client_id}_model.pth"
            torch.save(best_state, model_name)
    def domain_consistancy_detection(self,clients_cam,suspect_clients):
        suspect_clients = np.array(suspect_clients)
        suspect_indexes = [np.where(self.user_ids==i)[0].item() for i in suspect_clients]
        client_update_corrs = np.zeros((len(self.layers), len(self.user_ids)))
        global_update_corrs = np.zeros((len(self.layers), len(self.user_ids)))

        for indx, layer in enumerate(self.layers):
            layer_cams = clients_cam.clients_cam[indx].tolist()
            layer_cams.append(clients_cam.pre_global_cam[layer])
            result_imgs = np.argsort(np.argsort(layer_cams, axis=0), axis=0)
            client_update_corrs[indx] = get_sim(self.sort_sim, result_imgs[:-1],
                                                self.client_mean_imgs[layer][self.user_ids])  # +n_std*upper_limits['std'][indx]
            global_update_corrs[indx] = get_sim(self.sort_sim, np.repeat(np.expand_dims(result_imgs[-1],0),
                                                                         len(self.user_ids), axis=0),
                                                self.global_mean_imgs[layer][self.user_ids])  # +n_std*upper_limits['std'][indx]
        predict_values=[]
        slope_list=[]
        plt_score = []
        for indx, client_id in enumerate(self.user_ids):
            predict_value = self.kalmanfilter_predict_client(client_id, x=np.array(self.x[client_id])[:,-1])[:len(self.velocity)]
            # corr_scores.append(detect_helper.get_sim('sp',predict_value,client_update_corrs.T[indx]))
            predict_values.append(predict_value)

            differ=(client_update_corrs.T[indx] - predict_value)[int(len(self.velocity)/2):]
            z1 = np.polyfit(range(len(differ)), differ, 1)
            # 生成多项式对象
            p1 = np.poly1d(z1)
            pp1 = p1(range(len(differ)))
            slope_list.append(pp1[-1] - pp1[-2])
            plt_score.append(p1(range(len(differ))))

        predict_values = np.array(predict_values)
        slope_list = np.array(slope_list)

        certain_clients = suspect_clients[slope_list[suspect_indexes] < 0]
        cal_clients= np.array([i for i in self.user_ids if i not in certain_clients])
        cal_indexes = np.array([i for i,j in enumerate(self.user_ids) if j not in certain_clients])
        suspect_clients = np.array(list(set(suspect_clients) - set(certain_clients)))
        suspect_indexes = [np.where(cal_clients == i)[0].item() for i in suspect_clients]
        # print(f"Domain detection condition 1: {suspect_clients}")


        difference =  client_update_corrs.T[cal_indexes] - predict_values[cal_indexes]

        score = MadScore_mult(copy.deepcopy(difference))

        score_slope = MadScore_mult(slope_list[cal_indexes])

        # pre_attacker = set(suspect_clients)
        # 1-th condition
        pre_attacker = set(suspect_clients[np.sum(score > 2, axis=1)[suspect_indexes] > 0]) \
                       | set(suspect_clients[score_slope[suspect_indexes] > 1]) | set(
            suspect_clients[np.mean(difference[:, int(len(self.velocity) / 2):], axis=1)[suspect_indexes] > 0])

        return np.array(list(pre_attacker)),client_update_corrs.T, global_update_corrs.T
    def kalmanfilter_predict_client(self,client_id,x=None,z=None):
        filter = KalmanFilter(dim_x=len(x), dim_z=len(x), dim_u=int(len(x)/2))
        client_state_dict=self.clients_estimator[client_id].to('cpu').state_dict()
        # client_state_dict = torch.load(f'{self.filter_path}/pretrain_model_new.pth', map_location='cpu')
        # x_t-1
        filter.x = x
        filter.F = np.array(
            torch.cat((torch.cat((client_state_dict['k1.weight'], client_state_dict['k2.weight']), dim=1),
                       torch.cat((client_state_dict['k3'], client_state_dict['k4']), dim=1)), dim=0))

        filter.B = np.array(torch.cat((client_state_dict['k5.weight'],client_state_dict['k6']), dim=0))
        filter.H = np.array(torch.cat((torch.cat((client_state_dict['h1_and_v1.weight'], client_state_dict['h3']), dim=1),
                       torch.cat((client_state_dict['h4'], client_state_dict['h4_and_v2.weight']), dim=1)), dim=0))

        filter.R = np.cov(np.array(torch.cat((client_state_dict['h1_and_v1.bias'],client_state_dict['h4_and_v2.bias']), dim=0)))

        u_others_mean = np.mean(self.u_others[client_id],axis=1)
        W = np.concatenate((client_state_dict['w'], np.dot(client_state_dict['k7.weight'], u_others_mean)), axis=0)
        filter.Q = np.cov(W)
        filter.P = np.eye(len(x))
        filter.predict(u=np.array(client_state_dict['u']).flatten())
        filter.update(z)
        # x_t
        return filter.x+W
    def update_estimator(self,pre_attacker):
        normal_clients = [i for i, client in enumerate(self.user_ids) if client not in pre_attacker]
        local_sum = np.sum(self.local_updates_represenations[np.array(normal_clients)], axis=0)
        for index, client_id in enumerate(self.user_ids):
            if client_id not in pre_attacker:
                x_new = np.concatenate([self.local_updates_represenations[index], self.global_updates_represenations[index]])
                others = (local_sum - self.local_updates_represenations[index]) / (len(normal_clients) - 1)
                self.x[client_id] = np.concatenate([self.x[client_id], x_new.reshape(1, -1).T], axis=1)
                self.u_others[client_id] = np.concatenate([self.u_others[client_id], others.reshape(1, -1).T],
                                                          axis=1)
                if self.update_epoch:
                    loader = ClientDataLoader_update(self.x[client_id], self.u_others[client_id])
                    self.update_u(client_id, loader)
    def update_u(self,client_id,loader):
        estimator = self.clients_estimator[client_id]
        estimator.train()
        best_state = None
        lowest_loss = 10000
        for epoch in range(self.update_epoch):
            for i, (x_t_1, x_t, u_t_0) in enumerate(loader):
                x_t = x_t.cuda(self.device)
                x_t_1 = x_t_1.cuda(self.device)
                u_t_0 = u_t_0.cuda(self.device)

                output_x_t = estimator.update(x_t_1, u_t_0)
                loss = self.criterion(x_t, output_x_t)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                    best_state = copy.deepcopy(estimator.state_dict())

                # print(f"Training total loss: {loss.item()}")

        # model_name = f"{self.filter_path}/client{client_id}_model_new.pth"
        # torch.save(best_state, model_name)
        estimator.load_state_dict(best_state)
        self.clients_estimator[client_id] = estimator
    def detect(self,pre_attacker_tsd):
        print(f"Domain detection is starting...")
        pre_attacker_dcd, local_updates_represenations, global_updates_represenations = self.domain_consistancy_detection(
            self.clients_cam, pre_attacker_tsd)
        self.local_updates_represenations = local_updates_represenations
        self.global_updates_represenations = global_updates_represenations
        pre_attacker = set(pre_attacker_dcd)
        print(f"DDist anomaly clients are {pre_attacker}.")
        return pre_attacker