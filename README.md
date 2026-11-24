# COVAL

Code for **Counterfactual Occlusion-Aware Learning via Visibility Intervention
for LiDAR Anomaly Detection**, ICML 2026.

The latest code will be released soon ...

## Installation

The environment uses Python 3.9,
PyTorch 2.4.0 and TorchSparse 2.1.0.

```bash
conda env create -f environment.yml
conda activate coval
```

Please install **TorchSparse 2.1.0** following the
[TorchSparse installation instructions](https://github.com/mit-han-lab/torchsparse).


## Data

Please follow the official [STU dataset instructions](https://github.com/kumuji/stu_dataset)
for data download and directory layout. From the repository root, generate the
training metadata in `data/stu` and the anomaly bank in `data/anomaly`:

```bash
python tools/prepare_stu.py --dataset-root data/STU
```

## Pretrained Models

| Model | Weights |
|---|---|
| `coval` | [Baidu Netdisk](https://pan.baidu.com/s/1LeAX7UfwtACLGpcZ3QxJNA?pwd=b95v) (code: `b95v`) |

## Training

Run from the repository root. 

```bash
sh scripts/train.sh -g 1 -d stu \
    -c seg-minknet34 -n minknet34

sh scripts/train.sh -g 1 -d stu \
    -c seg-minknet34-syn -n minknet34-syn

sh scripts/train.sh -g 1 -d stu \
    -c seg-coval -n coval
```


Validation and the automatic post-training semantic test retain sequence 08
from SemanticKITTI. Use the separate test configurations below for anomaly
evaluation.

## Inference

Place the selected original checkpoint under its experiment's `model/` directory:

```bash
sh scripts/test.sh -g 1 -d stu \
    -c test-coval -n coval -w model_best
```


The anomaly sequence list is retained in `STUDataset._load_scenes`. Predictions
are restored to original scan point order and saved as:

```text
exp/stu/<experiment>/conf/
├── sequences_prediction/<sequence>/<frame>.npy   # (N,), uint8 class IDs
└── sequences_anomscore/<sequence>/<frame>.npy     # (N,), probability in channel 19
```


## Evaluation

Compute point-level AP, AUROC and FPR95, along with object-level metrics,
using the anomaly scores saved in `sequences_anomscore`:

```bash
python tools/compute_anom_metric.py \
    --data-path data/STU/validation \
    --conf-path exp/stu/coval/conf/sequences_anomscore \
    --output-path exp/stu/coval/metrics.json \
    --num-jobs 4
```


## Acknowledgements

This work is built upon [Pointcept](https://github.com/Pointcept/Pointcept) and
[TorchSparse](https://github.com/mit-han-lab/torchsparse), with STU data conventions
from [stu_dataset](https://github.com/kumuji/stu_dataset). 
