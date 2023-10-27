#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os
from tqdm import tqdm
from aug_method import *

class Data_augmentation:
    def __init__(self, graphs, ratio, sample, node_labels):

        self.graphs = graphs
        self.graph_num = len(self.graphs)
        self.sample = sample
        self.ratio = ratio
        self.node_labels = node_labels
        self.new_graphs = copy.deepcopy(graphs)

    def iteration(self):
        if self.sample == 'scaffold':
            graph_ = copy.deepcopy(self.graphs)
            it = tqdm(graph_)
            for inx, g in enumerate(it):
                valid = False
                try_num = 0
                num = g.number_of_edges()
                self.new_graphs[inx].graph['id'] = inx
                new_graph_id = self.graph_num + inx
                while valid is False and try_num <= 10:
                    new_g = scaffold_model(g, self.node_labels, rate=self.ratio)
                    if num != new_g.number_of_edges():
                        raise ValueError("number of edges error")
                    valid = True
                if valid is True:
                    final_new_g = copy.deepcopy(new_g)
                else:
                    final_new_g = copy.deepcopy(self.graphs[inx])
                final_new_g.graph['id'] = new_graph_id
                self.new_graphs.append(final_new_g)

        elif self.sample == 'group':
            graph_ = copy.deepcopy(self.graphs)
            it = tqdm(graph_)
            for inx, g in enumerate(it):
                valid = False
                try_num = 0
                num = g.number_of_edges()
                self.new_graphs[inx].graph['id'] = inx
                new_graph_id = self.graph_num + inx
                while valid is False and try_num <= 10:
                    new_g = group_model(g, self.node_labels, rate=self.ratio)
                    if num != new_g.number_of_edges():
                        raise ValueError("number of edges error")
                    valid = True
                if valid is True:
                    final_new_g = copy.deepcopy(new_g)
                else:
                    final_new_g = copy.deepcopy(self.graphs[inx])
                final_new_g.graph['id'] = new_graph_id
                self.new_graphs.append(final_new_g)

        return self.new_graphs






