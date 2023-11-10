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
```

## Experimental Details:

For all methods, we implement them using PyTorch 1.12.1 and Python 3.9.16. For Graph Transplant, NodeSam and four graph contrastive learning methods, we use the original codes from these papers with some necessary modifications. To provide a fair comparison, we use the same architecture of GNN backbones and the same training hyperparameters for all methods. The details of the setup are shown as follows. For the classification and regression tasks, we use five GNN models, including GCN, GIN, GAT, GraphSAGE and CMPNN. We set epoch as 100 and select the batch size from {32, 128, 256, 512}, the hidden size from {128, 256}, and the number of GNN layers from {3, 4, 5}. In addition, we use global mean pooling to obtain graph-level embeddings. During the training process, we apply the Adam optimizer with a learning rate of 0.001 to train all models.
