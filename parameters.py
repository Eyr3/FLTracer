import argparse
def parse_args():

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("--path", default="result/weights/epoch_202", type=str, help="The path of local updates need to be detected")
    parser.add_argument("--nocuda", default=False, type=bool, help="Use cpu when this is specified")

    # arguments for fl
    parser.add_argument("--clients_num", default=100, type=int, help="participants number")
    parser.add_argument('--model', default="resnet", type=str, help="model type: resnet, vgg, vit")
    parser.add_argument("--noniid", default=True, type=bool,help="whether in non-IID setting")

    # arguments for local anomaly detection
    parser.add_argument("--lambda_signv", default=3.0, type=float, help="signv threshold")
    parser.add_argument("--lambda_sortv", default=3.0, type=float, help="sortv threshold")
    parser.add_argument("--lambda_classv", default=2.5, type=float, help="classv threshold")

    # arguments for task detection
    parser.add_argument("--corr_detecting", default=False, type=bool, help="if true, use pearson correlation coefficient, otherwise use adjusted cosine similarity")
    parser.add_argument("--tau", default=-0.9, type=float, help="certain attacker threshold")
    parser.add_argument("--corr_first_num", default=1, type=int, help="suspect number")

    # arguments for domain detection
    parser.add_argument("--save_reference", default=True, type=bool,help="whether save reference")
    parser.add_argument("--reference_path", default="result/reference", type=str,help="The path of local updates as a reference")
    parser.add_argument('--reference_round', type=int, default=3, help="the number of average of multiple CAMs for each client")
    parser.add_argument("--epoch", default=5, type=int, help="epoch of estimating per client")
    parser.add_argument("--pretrain_epoch", default=100, type=int, help="pretrain epoch of estimating")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay of estimating")
    parser.add_argument("--lr", default=5e-8, type=float, help="learning rate per client")
    parser.add_argument("--pretrain_lr", default=2e-3, type=float, help="learning rate for pretrain")
    parser.add_argument("--save_model_path", default="result/model", type=str, help="Path of kalman filter estimator")
    parser.add_argument("--update_epoch", default=0, type=int, help="The number of rounds to update the estimator during detection")

    return parser.parse_args()

