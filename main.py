from detection.task_detecion import TaskDetection
from detection.domain_detecion import DomainDetection
from detection.local_anomaly_detecion import LocalAnomalyDetection
from utils.Client import Clients
from parameters import parse_args
from utils.CAM import ConvolutionAnomalyMatrix

if __name__ == "__main__":
    args = parse_args()
    print(f"Start detecting {args.path}...")
    local_updated_weights, user_ids = Clients(args.path).load_updates_from_files(-1)
    pre_attacker_lad = LocalAnomalyDetection(args,local_updated_weights,user_ids).detect()
    CAM = ConvolutionAnomalyMatrix(local_updated_weights, user_ids,noniid=args.noniid)
    candidates,pre_attacker_td = TaskDetection(args, CAM, user_ids).detect()
    pre_attacker_dd = DomainDetection(args, CAM, user_ids).detect(candidates)
    pre_attacker = pre_attacker_lad | pre_attacker_dd | pre_attacker_td

    print(f"Detection completed. Final attackers are {pre_attacker}.")


