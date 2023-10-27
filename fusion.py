#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
from copy import deepcopy
import logging
import warnings
from numpy import *
import numpy as np
import networkx as nx
from rdkit import Chem
from torch.optim import Adam
from rdkit import DataStructs
from model.G_fusion import build_model
from torch.nn.functional import softmax
from utils.MoleculeConvert import Graph2Mol
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class fusion:
    def __init__(self, args, idx_trains, idx_trains_aug, ori_data, scaffold_data, group_data,
                 ori_graphs, scaffold_graph, group_graph, device):

        self.model_fusion = build_model(args, device)
        path = './original/{}_{}_{}_{}_{}/{}/Model/original/'.format(args.bmname, args.lr_clf, args.split,
                                                                     args.hidden_dim, args.batch_size, args.model_name)
        file = "{}/{}_{}".format(path, args.bmname, args.seed)
        self.model_fusion.load_state_dict(torch.load(file, map_location=device))
        self.model_fusion.eval()

        params = filter(lambda p: p.requires_grad, self.model_fusion.parameters())
        self.optimizer = Adam(params, lr=args.lr_clf, weight_decay=5e-4)

        self.device = device
        self.batch_size = args.batch_size

        self.ori_data = [ori_data[idx] for idx in idx_trains]
        self.scaffold_data = [scaffold_data[idx] for idx in idx_trains_aug]
        self.group_data = [group_data[idx] for idx in idx_trains_aug]
        self.ori_graphs = [ori_graphs[idx] for idx in idx_trains]
        self.scaffold_graph = [scaffold_graph[idx] for idx in idx_trains_aug]
        self.group_graph = [group_graph[idx] for idx in idx_trains_aug]

    @torch.no_grad()
    def train(self, datas):
        self.model_fusion.train()
        idx = list(range(len(datas)))
        self.optimizer.zero_grad()
        x = datas[idx[0]]['x'].to(self.device)
        edge_index = datas[idx[0]]['edge_index'].to(self.device)
        edge_feat = datas[idx[0]]['edge_feat'].to(self.device)
        batch = torch.ones(len(datas[idx[0]]['x']), dtype=torch.int64).to(self.device) * 0
        y = datas[idx[0]]['y'].to(self.device)
        for i in range(1, len(idx)):
            batch = torch.hstack((batch, torch.ones(len(datas[idx[i]]['x']), dtype=torch.int64).to(self.device) * i))
            edge_index = torch.hstack((edge_index, datas[idx[i]]['edge_index'].to(self.device) + edge_index.max() + 1))
            edge_feat = torch.hstack((edge_feat, datas[idx[i]]['edge_feat'].to(self.device)))
            x = torch.vstack((x, datas[idx[i]]['x'].to(self.device)))
            y = torch.hstack((y, datas[idx[i]]['y'].to(self.device))).long()
        h_node, _ = self.model_fusion(x, edge_index, edge_feat, batch)
        batch_sizes = [torch.sum(batch == i).item() for i in range(torch.max(batch).item() + 1)]
        h_batch = torch.split(h_node, batch_sizes, dim=0)

        return h_node, h_batch

    def fusion_graph(self):
        aug_data = deepcopy(self.ori_data)

        # mixup
        for i, (ori_g, sca_g, gro_g) in enumerate(zip(self.ori_graphs, self.scaffold_graph, self.group_graph)):
            flag = 0
            try:
                ori_fp = Chem.RDKFingerprint(Graph2Mol(ori_g))
            except:
                flag = 1
            try:
                sca_fp = Chem.RDKFingerprint(Graph2Mol(sca_g))
                gro_fp = Chem.RDKFingerprint(Graph2Mol(gro_g))
                sim_os = DataStructs.FingerprintSimilarity(ori_fp, sca_fp)
                sim_og = DataStructs.FingerprintSimilarity(ori_fp, gro_fp)
                alpha_s = sim_os / (sim_os + sim_og)
                alpha_g = sim_og / (sim_os + sim_og)
                aug_data[i]['x'] = alpha_s * self.scaffold_data[i]['x'] + alpha_g * self.group_data[i]['x']
                mixed_adj = alpha_s * to_dense_adj(self.scaffold_data[i]['edge_index'])[0].double() + alpha_g * \
                            to_dense_adj(self.group_data[i]['edge_index'])[0].double()
                mixed_adj[mixed_adj < 0.5] = 0
                edge_index, _ = dense_to_sparse(mixed_adj)
                aug_data[i]['edge_index'] = edge_index
            except:
                pass

        ori_h, ori_batch = self.train(self.ori_data)
        aug_h, aug_batch = self.train(aug_data)

        for i, (h_ori, h_aug) in enumerate(zip(ori_batch, aug_batch)):
            distance_oa = F.cosine_similarity(h_ori, h_aug, dim=1)
            matrix_oa = softmax(distance_oa, dim=0)
            m_oa = matrix_oa.unsqueeze(1)
            self.ori_data[i]['x'] = (1 - m_oa) * self.ori_data[i]['x'] + m_oa * aug_data[i]['x']

        return self.ori_data


