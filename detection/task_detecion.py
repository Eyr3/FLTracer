from sklearn.decomposition import PCA
from utils.utils import *
from operator import itemgetter
class TaskDetection:
    def __init__(self,args,clients_cam,user_ids):
        self.clients_cam = clients_cam.clients_cam
        self.user_ids = user_ids
        self.model_name = args.model
        if 'vit' in self.model_name:
            self.pca = PCA(n_components=1)
        else:
            self.pca = PCA(n_components=2)
        self.certain_attacker_threshold = args.tau
        self.corr_first_num = args.corr_first_num
        self.corr_detecting = args.corr_detecting

    def calculate_task_similarity(self):
        task_similarity_scores = [[], []]
        if self.model_name == 'vit':
            limit_indx = 10000
            blocks = [f'blocks.{block_index}.' for block_index in range(9)]
            local_updated_weights = []
            for id, local_update in self.clients_cam.items():
                client_values = []
                for block in blocks:
                    layers = [key for key in local_update.keys() if block in key]
                    values = itemgetter(*layers)(local_update)
                    values = np.concatenate([value.flatten() for value in values])[:limit_indx]
                    client_values.append(values)
                local_updated_weights.append(client_values)

            local_updated_weights = np.array(local_updated_weights)
            scores = []
            for block_index in range(len(blocks)):
                value = local_updated_weights[:, block_index, :]

                score = self.pca.fit_transform(value).flatten()
                xmin = np.min(score)
                xmax = np.max(score)
                conv_weights_new_norm = (score - xmin) / (xmax - xmin)

                scores.append(conv_weights_new_norm)

            pca_values = np.array(scores).T
            imgs_median = np.median(pca_values, axis=0)
            task_similarity_scores[0].append(get_sim('adjcos_mul', pca_values, np.repeat(
                imgs_median.reshape(1, -1), len(pca_values), axis=0)))
            task_similarity_scores[1].append(get_sim('corr_mul', pca_values, np.repeat(
                imgs_median.reshape(1, -1), len(pca_values), axis=0)))
            return task_similarity_scores

        for layer_index, clients_cam in enumerate(self.clients_cam):
            imgs_median = np.median(clients_cam, axis=0)
            task_similarity_scores[0].append(get_sim('adjcos_mul', clients_cam, np.repeat(
                imgs_median.reshape(1, imgs_median.shape[0], imgs_median.shape[1]), len(clients_cam), axis=0)))
            task_similarity_scores[1].append(get_sim('corr_mul', clients_cam, np.repeat(
                imgs_median.reshape(1, imgs_median.shape[0], imgs_median.shape[1]), len(clients_cam), axis=0)))
        return task_similarity_scores

    def task_similarity_detection(self,scores):
        cos_scores = np.array(scores[0]).T
        corr_scores = np.array(scores[1]).T

        if self.corr_detecting:
            scores = corr_scores
        else:
            scores = cos_scores

        client_ids=np.array(self.user_ids)
        certain_attacker=client_ids[np.sum(cos_scores<self.certain_attacker_threshold,axis=1)>0]

        if self.model_name == 'vit':
            return [],certain_attacker


        certain_indexes = [np.where(client_ids == i)[0].item() for i in certain_attacker]
        detect_scores = np.delete(scores, certain_indexes, axis=0)
        detect_clients = np.delete(client_ids, certain_indexes)

        slope_list = []
        for index,score in enumerate(detect_scores):
            score = score[int(len(score)/2):]
            z1 = np.polyfit(range(len(score)), score, 1)
            p1 = np.poly1d(z1)
            pp1 = p1(range(len(score)))
            slope_list.append(pp1[-1] - pp1[-2])
        slope_score = np.array(slope_list)
        if self.corr_detecting:
            # pre_attacker = set(detect_clients[np.max(detect_scores, axis=1) <= 0])
            pre_attacker = (set(detect_clients[slope_score <= 0]) | set(
                detect_clients[np.max(detect_scores, axis=1) <= 0])) &  set(
                detect_clients[np.argsort(detect_scores, axis=0)[:self.corr_first_num].ravel()].tolist())
        else:
            # pre_attacker = set(detect_clients[np.max(detect_scores, axis=1) <= -0.5])
            pre_attacker = (set(detect_clients[np.argsort(slope_score)[:self.corr_first_num]]) & set(
            detect_clients[slope_score <= 0])) | set(detect_clients[np.max(detect_scores, axis=1) <= 0])

        # print(f"Task detection: {pre_attacker},{certain_attacker}")
        return list(pre_attacker),certain_attacker

    def detect(self):
        print(f"Task detection is starting...")
        task_similarity_scores = self.calculate_task_similarity()
        pre_attacker_tsd, certain_attacker = self.task_similarity_detection(task_similarity_scores)
        print(f"TSim anomaly candidates are {pre_attacker_tsd}, and {certain_attacker} are the certain attacker.")
        return pre_attacker_tsd, set(certain_attacker)
