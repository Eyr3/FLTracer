from sklearn.decomposition import PCA
import copy
from utils.utils import *
from operator import itemgetter
class LocalAnomalyDetection:
    def __init__(self,args,local_updated_weights,user_ids):
        self.local_updated_weights = local_updated_weights
        self.user_ids = user_ids
        self.fc_key = list(list(self.local_updated_weights.values())[0].keys())[-2:]
        # self.fc_key = ['classifier.weight', 'classifier.bias']  # ['linear.weight','linear.bias']
        self.client_weights_flattern = None
        self.sign_threshold = args.lambda_signv
        self.sort_threshold = args.lambda_sortv
        self.classifier_threshold = args.lambda_classv
    def processing_updates(self):
        layers = [key for key in self.local_updated_weights[self.user_ids[0]].keys()]
        client_weights = np.zeros((len(self.user_ids), 1))
        for layer in layers:
            layer_weights = []
            for user_id in self.user_ids:
                client_update = self.local_updated_weights[user_id][layer]
                client_value = client_update.ravel()
                layer_weights.append(client_value)
            client_weights = np.concatenate((client_weights, np.array(layer_weights)), axis=1)

        self.client_weights_flattern = client_weights[:, 1:]
    def detecting_expert_1_2(self,threshold=3,detecting_method=None):
        client_weights = copy.deepcopy(self.client_weights_flattern)
        size_th = 100000
        pca = PCA(n_components=1)
        n, d = client_weights.shape
        if d > size_th:
            idx = np.sort(np.random.choice(d, size_th, replace=False))
            client_weights = client_weights[:, idx]

        if detecting_method == 'sign':
            clients_sort = np.sign(np.array(client_weights))
        else:
            clients_sort = np.argsort(np.argsort(np.array(client_weights), axis=0), axis=0)

        clients_sort = clients_sort - np.mean(clients_sort, 0)
        sore_list = pca.fit_transform(clients_sort)
        scores=MadScore(sore_list)#1.6 sign  10 add noise
        return set(np.array(self.user_ids)[(scores>threshold).ravel()])
    def detecting_expert_3(self,threshold=2):
        layer_weights=[]
        
        for client_id in self.user_ids:
            tmp = itemgetter(*self.fc_key)(self.local_updated_weights[client_id])
            layer_weights.append(np.concatenate([a.ravel() for a in tmp]))

        X = np.array(layer_weights)
        pca = PCA(n_components=1)
        pca.fit(X)
        X_new = pca.transform(X)
        # inv_cov_matrix = np.linalg.inv(np.cov(X_new, rowvar=False))
        # mean_distr = X_new.mean(axis=0)
        # values = np.array([mahalanobis(value, mean_distr, inv_cov_matrix) for value in X_new])
        values=X_new.ravel()
        score = MadScore(values)
        pre_attacker=set(np.array(self.user_ids)[abs(score)>threshold])
        return pre_attacker
    def detect(self):
        print("Local Anomaly Detection is starting...")
        self.processing_updates()
        pre_attacker1 = self.detecting_expert_1_2(threshold=self.sign_threshold,detecting_method='sign')
        print(f"signv anomaly clients are {pre_attacker1}")
        pre_attacker2 = self.detecting_expert_1_2(threshold=self.sort_threshold, detecting_method='sort')
        print(f"sortv anomaly clients are {pre_attacker2}")
        pre_attacker3 = self.detecting_expert_3(threshold=self.classifier_threshold)
        print(f"classv anomaly clients are {pre_attacker3}")
        return set(pre_attacker1) | set(pre_attacker2) | set(pre_attacker3)
