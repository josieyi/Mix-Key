#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn.conv import GINConv, GCNConv, SAGEConv, GATConv
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.models import MLP


class GIN(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.args = args
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(),
                                                      Linear(hidden_channels, hidden_channels), ReLU(),
                                                      BatchNorm1d(hidden_channels)), train_eps=False))
            else:
                self.layers.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(),
                                                      Linear(hidden_channels, hidden_channels), ReLU(),
                                                      BatchNorm1d(hidden_channels)), train_eps=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, out_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mlp1(x)
        last = x
        x = F.log_softmax(x, -1)
        x_soft = F.softmax(x, -1)
        return x, x_soft, last


class GCN(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.args = args
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(GCNConv(in_channels, hidden_channels, bias=False))
            else:
                self.layers.append(GCNConv(hidden_channels, hidden_channels, bias=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, out_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=None)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.mlp1(x)
        last = x
        x_log = F.log_softmax(x, -1)
        x_sig = torch.sigmoid(x)
        return x_log, x_sig, last

class GAT(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(GATConv(in_channels, hidden_channels//4, heads=4, bias=False))
            else:
                self.layers.append(GATConv(hidden_channels, hidden_channels//4, heads=4, bias=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, out_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.mlp1(x)
        last = x
        x = F.log_softmax(x, -1)
        x_soft = F.softmax(x, -1)
        return x, x_soft, last

class GraphSAGE(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.args = args
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(0, num_layers):
            if layer == 0:
                self.layers.append(SAGEConv(in_channels, hidden_channels, bias=False))
            else:
                self.layers.append(SAGEConv(hidden_channels, hidden_channels, bias=False))

        self.mlp1 = MLP([hidden_channels, hidden_channels, out_channels], dropout=self.dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.mlp1(x)
        last = x
        x_log = F.log_softmax(x, -1)
        x_sig = torch.sigmoid(x)
        return x_log, x_sig, last

def build_model(args, device):
    if args.model_name == 'GCN':
        return GCN(args=args, in_channels=args.feat_dim, hidden_channels=args.hidden_dim, out_channels=args.output_dim,
                   num_layers=args.num_layer, dropout=args.dropout).to(device)
    elif args.model_name == 'GIN':
        return GIN(args=args, in_channels=args.feat_dim, hidden_channels=args.hidden_dim, out_channels=args.output_dim,
                   num_layers=args.num_layer, dropout=args.dropout).to(device)
    elif args.model_name == 'GAT':
        return GAT(args=args, in_channels=args.feat_dim, hidden_channels=args.hidden_dim, out_channels=args.output_dim,
                   num_layers=args.num_layer, dropout=args.dropout).to(device)
    elif args.model_name == 'GraphSAGE':
        return GraphSAGE(args=args, in_channels=args.feat_dim, hidden_channels=args.hidden_dim, out_channels=args.output_dim,
                         num_layers=args.num_layer, dropout=args.dropout).to(device)
