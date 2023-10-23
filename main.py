from detection.task_detecion import TaskDetection
from detection.domain_detecion import DomainDetection
from detection.local_anomaly_detecion import LocalAnomalyDetection
from utils.Client import Clients
from utils.parameters import Params
import yaml
import argparse
from utils.CAM import ConvolutionAnomalyMatrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection')
    parser.add_argument('--params', dest='params', default='config/dirty_label_detect.yaml')  #

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = Params(**params)
    args.check()
    print(f"Start detecting {args.path}...")
    local_updated_weights, user_ids = Clients(args.path).load_updates_from_files(-1)
    pre_attacker_lad = LocalAnomalyDetection(args,local_updated_weights,user_ids).detect()
    CAM = ConvolutionAnomalyMatrix(local_updated_weights, user_ids,noniid=args.noniid)
    candidates,pre_attacker_td = TaskDetection(args, CAM, user_ids).detect()
    if args.reference_path is not None:
        pre_attacker_dd = DomainDetection(args, CAM, user_ids).detect(candidates)
        pre_attacker = pre_attacker_lad | pre_attacker_dd | pre_attacker_td
    else:
        pre_attacker = pre_attacker_lad | pre_attacker_td

    print(f"Detection completed. Final attackers are {pre_attacker}.")


