# Mix-Key: Graph Mixup with Key Structures for Molecular Property Prediction

## Framework

![image](https://github.com/josieyi/Mix-Key/blob/main/figures/Mix-Key%20Framework.jpg)

## Usage:

1. Train original model
```
python train.py --bmname {dataset name} --train_type train --task_type {reg/class}
```
2. Molecular Property Prediction
```
python train.py --bmname {dataset name} --train_type retrain --task_type {reg/class}
