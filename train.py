#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
import logging
import warnings
from numpy import *
import numpy as np
import networkx as nx
from rdkit import Chem
import torch.nn.functional as F
from fusion import *
from configs import arg_parse_clf
from model.GNNs import build_model
from torch.optim import Adam
from copy import deepcopy
from utils.data_process import read_data, Network2TuDataset, data_split
from utils.data_augmentation import Data_augmentation
from torch.utils.data.sampler import BatchSampler
from evaluate import eval_rocauc, eval_rmse

warnings.filterwarnings('ignore')

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(args, model, optimizer, datas, batch_size, device):
    model.train()
    loss = 0
    epochs = int(np.ceil(len(datas) / batch_size))
    ids = list(range(len(datas)))
    np.random.shuffle(ids)
    idx_all = list(BatchSampler(ids, batch_size, drop_last=False))
    for epoch in range(epochs):
        idx = idx_all[epoch]
        optimizer.zero_grad()
        x = datas[idx[0]]['x'].to(device)
        edge_index = datas[idx[0]]['edge_index'].to(device)
        edge_feat = datas[idx[0]]['edge_feat'].to(device)
        batch = torch.ones(len(datas[idx[0]]['x']), dtype=torch.int64).to(device) * 0
        y = datas[idx[0]]['y'].to(device)
        for i in range(1, len(idx)):
            batch = torch.hstack((batch, torch.ones(len(datas[idx[i]]['x']), dtype=torch.int64).to(device) * i))
            try:
                edge_index = torch.hstack((edge_index, datas[idx[i]]['edge_index'].to(device) + edge_index.max() + 1))
            except:
                edge_index = torch.hstack((edge_index, datas[idx[i]]['edge_index'].to(device) + 1))
            edge_feat = torch.hstack((edge_feat, datas[idx[i]]['edge_feat'].to(device)))
            x = torch.vstack((x, datas[idx[i]]['x'].to(device)))
            y = torch.vstack((y, datas[idx[i]]['y'].to(device))).long()
        _, _, out = model(x, edge_index, edge_feat, batch)
        y1 = np.array(y.detach().cpu(), dtype=np.float)
        y1[y1 < 0] = np.nan
        is_labeled = y1 == y1
        if args.task_type == 'class':
            try:
                loss = cls_criterion(out.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            except:
                out1 = out.to(torch.float32)[0][is_labeled[0]]
                for i in range(1, len(is_labeled)):
                    o = out.to(torch.float32)[i][is_labeled[i]]
                    out1 = torch.cat((out1, o))
                y1 = y.to(torch.float32)[0][is_labeled[0]]
                for i in range(1, len(is_labeled)):
                    y_ = y.to(torch.float32)[i][is_labeled[i]]
                    y1 = torch.cat((y1, y_))
                loss = cls_criterion(out1, y1)
        else:
            loss = reg_criterion(out.to(torch.float32), y.to(torch.float32))
        loss.backward()
        optimizer.step()
    return model, optimizer, loss


@torch.no_grad()
def test(args, model, datas, batch_size, device):
    model.eval()
    y_true = []
    y_pred = []
    loss = 0
    epochs = int(np.ceil(len(datas) / batch_size))
    ids = list(range(len(datas)))
    idx_all = list(BatchSampler(ids, batch_size, drop_last=False))
    for epoch in range(epochs):
        idx = idx_all[epoch]
        x = datas[idx[0]]['x'].to(device)
        edge_index = datas[idx[0]]['edge_index'].to(device)
        edge_feat = datas[idx[0]]['edge_feat'].to(device)
        batch = torch.ones(len(datas[idx[0]]['x']), dtype=torch.int64).to(device) * 0
        y = datas[idx[0]]['y'].to(device)
        for i in range(1, len(idx)):
            batch = torch.hstack((batch, torch.ones(len(datas[idx[i]]['x']), dtype=torch.int64).to(device) * i))
            try:
                edge_index = torch.hstack((edge_index, datas[idx[i]]['edge_index'].to(device) + edge_index.max() + 1))
            except:
                edge_index = torch.hstack((edge_index, datas[idx[i]]['edge_index'].to(device) + 1))
            edge_feat = torch.hstack((edge_feat, datas[idx[i]]['edge_feat'].to(device)))
            x = torch.vstack((x, datas[idx[i]]['x'].to(device)))
            y = torch.vstack((y, datas[idx[i]]['y'].to(device))).long()
        _, _, out = model(x, edge_index, edge_feat, batch)
        y1 = np.array(y.detach().cpu(), dtype=np.float)
        y1[y1 < 0] = np.nan
        is_labeled = y1 == y1
        if args.task_type == 'class':
            try:
                loss = cls_criterion(out.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            except:
                out1 = out.to(torch.float32)[0][is_labeled[0]]
                for i in range(1, len(is_labeled)):
                    o = out.to(torch.float32)[i][is_labeled[i]]
                    out1 = torch.cat((out1, o))
                y1 = y.to(torch.float32)[0][is_labeled[0]]
                for i in range(1, len(is_labeled)):
                    y_ = y.to(torch.float32)[i][is_labeled[i]]
                    y1 = torch.cat((y1, y_))
                loss = cls_criterion(out1, y1)
            y_true.append(y.detach().cpu())
        else:
            loss = reg_criterion(out.to(torch.float32), y.to(torch.float32))
            if y.size().numel() == 1:
                y_true.append(y.detach().cpu().unsqueeze(dim=1))
            else:
                y_true.append(y.detach().cpu())
        y_pred.append(out.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']

    return loss, result


def model_train(args, dataset, idx_train, idx_val, idx_test, device):
    model = build_model(args, device)
    lr = args.lr_clf

    if args.train_type == 'retrain':
        path = "./Experiments/{}_{}_{}_{}_{}/{}_{}/Model/Augmentation_{}_{}/".format(args.bmname, args.lr_clf,
                                                                                     args.split, args.hidden_dim,
                                                                                     args.ratio, args.model_name,
                                                                                     args.batch_size, args.sample,
                                                                                     args.num)
        file = "{}/{}_{}_retrain".format(path, args.bmname, args.seed)
    else:
        path = './original/{}_{}_{}_{}_{}/{}/Model/original/'.format(args.bmname, args.lr_clf, args.split,
                                                                     args.hidden_dim, args.batch_size, args.model_name)
        file = "{}/{}_{}".format(path, args.bmname, args.seed)

    if not os.path.exists(path):
        os.makedirs(path)
    params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer setting
    optimizer = Adam(params, lr=lr, weight_decay=5e-4)
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_patience, gamma=args.lr_gamma)

    # data split
    train_dataset = [dataset[idx] for idx in idx_train]
    val_dataset = [dataset[idx] for idx in idx_val]
    test_dataset = [dataset[idx] for idx in idx_test]

    # model train
    t_total = time.time()
    train_loss_record = []
    val_result_record = []
    val_loss_record = []
    epochs = args.epochs
    best_result = None
    for epoch in range(1, epochs + 1):
        t = time.time()
        model, optimizer, loss_train = train(args, model, optimizer, train_dataset, args.batch_size, device)
        # lr decay
        if args.lr_decay:
            scheduler.step()
        loss_val, result_val = test(args, model, val_dataset, args.batch_size, device)
        if best_result is None:
            best_result = result_val
            torch.save(model.state_dict(), file)
        elif (args.task_type == 'class' and result_val > best_result) or (args.task_type == 'reg' and result_val < best_result):
            best_result = result_val
            torch.save(model.state_dict(), file)
            logging.info('Epoch:{:04d},train(loss:{:.4f};val(loss:{:.4f},result:{:.4f}),lr:{:.7f},time:{:.4f}s'
                         .format(epoch, loss_train.item(), loss_val.item(), result_val,
                                 optimizer.state_dict()['param_groups'][0]['lr'], time.time() - t))

        val_result_record.append(result_val)
        val_loss_record.append(loss_val.cpu().detach())
        train_loss_record.append(loss_train.cpu().detach())
    logging.info("Optimization Finished! Total time elapsed: {:.4f}s".format(time.time() - t_total))

    model.load_state_dict(torch.load(file, map_location=device))  # load weight
    loss_test, result_test = test(args, model, test_dataset, args.batch_size, device)
    logging.info("loss test: {}".format(loss_test))
    logging.info("result test: {}".format(result_test))
    logging.info(f'Model saved at {file}')

    return result_test, model


def main(args):
    if args.train_type == 'retrain':
        logs_dir = './Experiments/{}_{}_{}_{}_{}/{}_{}/logs/Augmentation/'.format(args.bmname, args.lr_clf, args.split,
                                                                                  args.hidden_dim, args.ratio,
                                                                                  args.model_name, args.batch_size)
        logs_file = logs_dir + "{}_{}_{}_retrain.log".format(args.seed, args.batch_size, args.lr_decay)
    else:
        logs_dir = './original/{}_{}_{}_{}_{}/{}/logs/original/'.format(args.bmname, args.lr_clf, args.split,
                                                                        args.hidden_dim, args.batch_size,
                                                                        args.model_name)
        logs_file = logs_dir + "OriTrain_{}.log".format(args.seed)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logging.basicConfig(level=logging.INFO,
                        filename=logs_file,
                        filemode="w",
                        format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if (torch.cuda.is_available() and args.cuda):
        print('cuda:{}'.format(args.gpu))
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
        logging.info("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        logging.info("Device set to : cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('seed:{}'.format(args.seed))

    # load data
    logging.info('Load data ...')
    graphs, args.max_num_nodes, node_map, task_num = read_data(args.datadir, args.bmname)
    dataset = [Network2TuDataset(graph, device) for graph in graphs]
    args.feat_dim = len(node_map)
    args.output_dim = int(task_num)

    # data information
    logging.info('the number of graphs:{}'.format(len(dataset)))
    logging.info('the number of class:{}'.format(args.output_dim))
    logging.info('the dimension of feat:{}'.format(args.feat_dim))
    logging.info('the max number of nodes:{}'.format(args.max_num_nodes))

    # data split
    train_splits, test_splits, val_splits = data_split(args, dataset)

    if args.train_type != 'train':
        aug_models = ['scaffold', 'group']
        aug_data_all = []
        aug_graph_all = []
        for aug_model in aug_models:
            all_aug_dataset = []
            all_graphs = []
            args.sample = aug_model
            # aug data
            for num in range(args.iter):
                ori_dataset = deepcopy(dataset)
                ori_graphs = deepcopy(graphs)
                aug_dir = './Experiments/Data/{}_AUG_{}/{}/{}/'.format(args.bmname, args.ratio, args.sample, num)
                args.num = num
                os.makedirs(aug_dir)
                logging.info('Data augmentation ...')
                for graph in graphs:
                    graph.graph['stand'] = 1
                data_augment = Data_augmentation(graphs=graphs, ratio=args.ratio,
                                                 sample=args.sample, node_labels=node_map)
                gs = data_augment.iteration()
                gs = gs[len(graphs):]

                ori_graphs.extend(gs)
                aug = [Network2TuDataset(graph, device) for graph in gs]
                for data_i in range(len(aug)):
                    aug[data_i]['y'] = ori_dataset[data_i]['y']
                ori_dataset.extend(aug)
                aug_graphs = deepcopy(ori_graphs)
                aug_dataset = deepcopy(ori_dataset)
                all_aug_dataset.append(aug_dataset)
                all_graphs.append(aug_graphs)
            aug_data_all.append(all_aug_dataset)
            aug_graph_all.append(all_graphs)

        for num in range(args.iter):
            args.num = num
            idx_trains = torch.LongTensor(train_splits).to(device)
            idx_vals = torch.LongTensor(val_splits).to(device)
            idx_tests = torch.LongTensor(test_splits).to(device)
            idx_train_aug = [i + len(graphs) for i in idx_trains]

            # feature fusion
            scaffold_dataset = aug_data_all[0][num]
            group_dataset = aug_data_all[1][num]
            scaffold_graph = aug_graph_all[0][num]
            group_graph = aug_graph_all[1][num]

            fusion_model = fusion(args, idx_trains, idx_train_aug, dataset, scaffold_dataset, group_dataset,
                                  graphs, scaffold_graph, group_graph, device)
            final_dataset = fusion_model.fusion_graph()
            all_final_dataset = dataset + final_dataset
            
            idx_trains_aug = [i + len(dataset) for i in range(len(final_dataset))]
            idx_trains = torch.hstack((idx_trains, torch.tensor(idx_trains_aug).to(device))).to(device)

            logging.info('train model ...')
            result, model = model_train(args, all_final_dataset, idx_trains, idx_vals, idx_tests, device)
            logging.info("test_result:" + str(result) + "\n")
            print(("test_result:" + str(result) + "\n"))
    else:
        logging.info('train model ...')
        idx_trains = torch.LongTensor(train_splits).to(device)
        idx_vals = torch.LongTensor(val_splits).to(device)
        idx_tests = torch.LongTensor(test_splits).to(device)
        result, _ = model_train(args, dataset, idx_trains, idx_vals, idx_tests, device)
        logging.info("original_test_result:" + str(result) + "\n")
        print("original_test_result:" + str(result) + "\n")


if __name__ == '__main__':
    args = arg_parse_clf()
    main(args)
