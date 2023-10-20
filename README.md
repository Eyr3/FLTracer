# FLTracer
FLTracer: Accurate Poisoning Attack Provenance in Federated Learning

This is the Additional Experimental Results for FLTracer ([PDF](https://github.com/Eyr3/FLTracer/blob/main/FLTracer_Additional_Experimental_Results.pdf)).

## Usage
### Prepare the dataset
- Image Classification: MNIST, EMNIST, CIFAR10,
- Traffic Sign Classification: German Traffic Sign Recognition Benchmark (GTSRB),
- human Activity Recognition: HAR
- Object Classification: [BDD100K dataset](https://www.vis.xyz/bdd100k/)

### Installation
Our code is implemented and evaluated on pytorch. The following packages are used by our code.

- `torch==2.0.1`
- `numpy==1.24.3`
- `scikit-learn==1.3.1`
- `filterpy==1.4.5`
- `tqdm==4.66.1`

Our code is evaluated on `Python 3.8.11`.

### Saving Updates
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

### Anomaly Detection
After saving local updates, you can detect anomalies with various settings using the following command:

```
python main.py  --path result/weights/epoch_xxx                   \
                --clients_num 100                                 \
                --model resnet,vgg,vit                            \
                --noniid  True,False                              \
                --lambda_signv 3.0                                \
                --lambda_sortv 3.0                                \
                --lambda_classv 3.0                               \
                --tau -0.9                                        \
                --corr_first_num 1                                \
                --save_reference   True,False                     \
                --reference_path result/reference                 \
                --reference_round 3                               \
                --pretrain_epoch 100                              \
                --epoch 5                                         \
                --update_epoch 0                                  \

                (optional - checkpoint)
                --save_reference   True,False                     \
                --save_model_path result/model                    \
```
- `path` : is should be matched with your path to be detected. 
- `lambda_signv`, `lambda_sortv`, `lambda_classv`: are thresholds in Local Anomaly Detection. We use MAD to detect anomalies, with default settings of 2.5, 3.0, or higher.
- `tau` : is the threshold in Task Detection, with default settings of -0.9.
- `reference_path` : is should be matched with your reference path for Domain Detection. 
- `update_epoch` : is the epoch of updating the Kalman Filter estimator while detecting.

### Repeating Experiments
- run experiments for the four datasets:
  ```python xxxxxx.py --params utils/X.yaml
  ```
- 
note - https://github.com/ebagdasa/backdoors101


## Citation
``` 123
```

## Acknowledgements
[https://github.com/ebagdasa/backdoors101](https://github.com/ebagdasa/backdoors101)
[https://github.com/AI-secure/DBA](https://github.com/AI-secure/DBA)

