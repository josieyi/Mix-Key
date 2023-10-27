# Mix-Key: Graph Mixup with Key Structures for Molecular Property Prediction

## Framework

<div align='center'>
<p><img src="figures/Mix-Key Framework.jpeg" width='500' /></p>
</div>

## Usage:

1. Train original model
```
python train.py --bmname {dataset name} --train_type train --task_type {reg/class}
```
2. Molecular Property Prediction
```
python train.py --bmname {dataset name} --train_type retrain --task_type {reg/class}