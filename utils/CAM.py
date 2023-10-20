import copy

import numpy as np
from sklearn.decomposition import PCA
from utils.utils import *
from operator import itemgetter
from scipy.spatial.distance import mahalanobis
class ConvolutionAnomalyMatrix:
    def __init__(self,local_updated_weights,user_ids,pre_global_cam=None,noniid=True,measurement=False):
        self.noniid = noniid
        self.measurement = measurement
        self.local_updated_weights = local_updated_weights
        self.user_ids = user_ids
        self.pre_global_cam = pre_global_cam
        self.limit_indx = 32
        self.pca = PCA(n_components=2)
        self.layers = [key for key in list(self.local_updated_weights.values())[0].keys() if
                       ('layer' in key and 'conv' in key) or ('features' in key and 'weight' in key)]
        # if 'vgg' in self.model_name:
        #     self.layers.sort(key=detect_helper.sort_key)  #
        # else:
        #     self.layers.sort()

        self.global_cam = dict()
        # for layer in self.layers:
        #     self.global_imgs[layer] = np.load(f'{self.no_attack_path}/{layer}/epoch200_global.npz', allow_pickle=True)[
        #         'arr_0'].tolist()
        self.clients_cam = self.calculate_cam()
    def calculate_cam(self):
        clients_cam=[]
        local_updated_weights = []

        # # # have convolution kernel
        for user_id in self.user_ids:
            client_update = self.local_updated_weights[user_id]
            local_updated_weights.append(itemgetter(*self.layers)(client_update))
        for layer_index,layer in enumerate(self.layers):
            tmp_value = np.array([*np.array(local_updated_weights, dtype=object)[:, layer_index]], dtype=object)[:,
                        :self.limit_indx, :self.limit_indx, :, :]
            if self.measurement:
                # global update std
                normal_std = np.std(tmp_value[-1]) / 10
                tmp_value = tmp_value + np.random.normal(0, normal_std, tmp_value.shape)
            client_value = tmp_value.reshape(tmp_value.shape[0], tmp_value.shape[1], tmp_value.shape[2], -1)
            # layer_weights.append(client_value)
            (x0, x1, x2, _) = client_value.shape
            imgs = np.zeros([x0, x1, x2])
            for j in range(x1):
                for k in range(x2):
                    conv_weights = client_value[:, j, k, :]

                    conv_weights_new = self.pca.fit_transform(conv_weights)
                    if not self.noniid:
                        conv_weights_new = abs(conv_weights_new)
                    else:
                        inv_cov_matrix = np.linalg.inv(np.cov(conv_weights_new, rowvar=False))
                        mean_distr = conv_weights_new.mean(axis=0)
                        conv_weights_new = np.array(
                            [mahalanobis(value, mean_distr, inv_cov_matrix) for value in conv_weights_new])

                    imgs[:, j, k] = norm1(conv_weights_new)

            clients_cam.append(imgs)

            # img_0 = copy.deepcopy(self.global_imgs[layer])
            # imgs = np.concatenate((imgs,np.expand_dims(np.array(img_0),axis=0)),axis=0)
            self.global_cam[layer] = np.mean(imgs, axis=0)
        return clients_cam