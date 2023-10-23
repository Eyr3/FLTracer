# FLTracer
FLTracer: Accurate Poisoning Attack Provenance in Federated Learning

This is the Additional Experimental Results for FLTracer ([PDF](https://github.com/Eyr3/FLTracer/blob/main/FLTracer_Additional_Experimental_Results.pdf)).


## Installation
Our code is implemented and evaluated on pytorch. The following packages are used by our code.

- `torch==2.0.1`
- `numpy==1.24.3`
- `scikit-learn==1.3.1`
- `filterpy==1.4.5`
- `tqdm==4.66.1`

Our code is evaluated on `Python 3.8.11`.


## Usage
### Prepare Datasets
- Image Classification: MNIST, EMNIST, CIFAR10,
- Traffic Sign Classification: German Traffic Sign Recognition Benchmark (GTSRB),
- Human Activity Recognition: HAR
- Object Classification: [BDD100K dataset](https://www.vis.xyz/bdd100k/)


### Save Updates
To detect attackers, you should provide local updates of suspect clients, which need to have the following folder structure. 
```shell
results
|-- weights < updates need to be detected >
    |-- epoch_xxx < local updates at xxx epoch >
        |-- node_yyy.npz  < local updates of yyy client >
        |-- ...
    |-- ...
|-- reference < reference updates for domain detection >
    |-- epoch_xxx < local updates at xxx epoch >
        |-- node_yyy.npz  < local updates of yyy client >
        |-- ...
    |-- ...
|-- model
    < contains all saved models >
```

### Detect Anomalies
After saving local updates, you can detect anomalies with various settings using the following command:

```
python main.py  --params config/add_noise_detect.yaml
```
The following is a description of some parameters in the configuration file:
- `path` : is should be matched with your path to be detected.
- `model`: is the model type of local updates, which support resnet, vgg, and vit.
- `clients_num`: is the number of participating clients.
- `lambda_signv`, `lambda_sortv`, `lambda_classv`: are thresholds in Local Anomaly Detection. We use MAD to detect anomalies, with default settings of 2.5, 3.0, or higher.
- `tau` : is the threshold in Task Detection, with default settings of -0.9.
- `corr_first_num`: is the number of malicious clients, if unknown, please set the same value as `clients_num`.
- `reference_path` : is should be matched with your reference path for Domain Detection. 
- `update_epoch` : is the epoch of updating the Kalman Filter estimator while detecting.

### Repeat Experiments
We provide all experimental examples on CIFAR10 ResNet. Run experiments for add noise attack detection:
```python main.py --params config/add_noise_detect.yaml ```
Run experiments for dirty label attack detection:
```python main.py --params config/dirty_label_detect.yaml ```
Run experiments for sign flipping attack detection:
```python main.py --params config/sign_flipping_detect.yaml ```
Run experiments for adaptive untargeted attack detection, e.g. MB attack and Fang attack:
```python main.py --params config/adaptive_untarget_attack_detect.yaml ```
Run experiments for backdoor attack detection:
```python main.py --params config/patch_BN_backdoor_detect.yaml ```

In our experiments, we use [backdoors101](https://github.com/ebagdasa/backdoors101) to train and attack models to prepare the detecting data.

## Citation
``` 123
```


