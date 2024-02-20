# Mix-Key: Graph Mixup with Key Structures for Molecular Property Prediction

## Framework

![image](https://github.com/josieyi/Mix-Key/blob/main/figures/Mix-Key%20Framework.jpg)

## Environment:

We used the following Python packages for core development. We tested on `Python 3.9`.
```
- pytorch 1.11.0
- torch-geometric 2.2.0
```

## Datasets:

The datasets used for the experiments are provided in the `data` directory of this repository.

## Usage:

1. Save generated isomers
```
python save_data.py --bmname {dataset name}
```
2. Train original model
```
python train.py --bmname {dataset name} --train_type train --task_type {reg/class}
```
3. Molecular Property Prediction
```
python train.py --bmname {dataset name} --train_type retrain --task_type {reg/class}
```

