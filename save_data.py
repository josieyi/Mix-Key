#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
import warnings
import numpy as np
import networkx as nx
from rdkit import Chem
from fusion import *
from configs import arg_parse_clf
from copy import deepcopy
from openpyxl import Workbook
from utils.MoleculeConvert import Graph2Mol
from utils.data_process import read_data, Network2TuDataset
from utils.data_augmentation import Data_augmentation

warnings.filterwarnings('ignore')

def Graph_2Smile(args, graphs):
    f = Workbook()  # create workbook
    sheet1 = f.create_sheet('{}'.format(args.bmname))
    try:
        if len(graphs[0].graph['label']) != 1:
            name_list = ['smiles']
            for i in range(len(graphs[0].graph['label'])):
                name_list.append('label_{}'.format(i))
            sheet1.append(name_list)
        else:
            sheet1.append(['smiles', 'label'])
    except:
        sheet1.append(['smiles', 'label'])
    f.remove(f['Sheet'])
    excel_name = '{}_{}'.format(args.bmname, args.sample) + '.xlsx'
    Data_dir = './Experiments/Data/{}_AUG_{}/{}/'.format(args.bmname, args.ratio, args.sample)
    if os.path.exists(Data_dir) == False:
        os.makedirs(Data_dir)
    excel_name = os.path.join(Data_dir, excel_name)
    f.save(excel_name)
    for index, graph in enumerate(graphs):
        mol = Graph2Mol(graph)
        if mol == None:
            logging.info('The {}th mol is not standardized')
            continue
        smiles = Chem.MolToSmiles(mol)
        try:
            if len(graph.graph['label']) == 1:
                label = graph.graph['label']
                sheet1.append([smiles, label])
            else:
                new_list = [smiles]
                label = graph.graph['label']
                new_list = new_list + label
                sheet1.append(new_list)
        except:
            label = graph.graph['label']
            sheet1.append([smiles, label])
        f.save(excel_name)

def main(args):

    if (torch.cuda.is_available() and args.cuda):
        print('cuda:{}'.format(args.gpu))
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load data
    graphs, args.max_num_nodes, node_map, task_num = read_data(args.datadir, args.bmname)

    # data augment
    aug_models = ['scaffold', 'group']
    for aug_model in aug_models:
        args.sample = aug_model
        for graph in graphs:
            graph.graph['stand'] = 1
        data_augment = Data_augmentation(graphs=graphs, ratio=args.ratio,
                                         sample=args.sample, node_labels=node_map)
        gs = data_augment.iteration()
        Graph_2Smile(args, gs[len(graphs):])



if __name__ == '__main__':
    args = arg_parse_clf()
    main(args)
