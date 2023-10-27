#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def arg_parse_clf():
    parser = argparse.ArgumentParser(description='Training arguments.')

    parser.add_argument('--task_type', dest='task_type', help='task type:class, reg')
    parser.add_argument('--datadir', dest='datadir', help='data direction')
    parser.add_argument('--bmname', dest='bmname', help='data name')
    parser.add_argument('--task', dest='task', help='classification task')
    parser.add_argument('--feat_dim', type=int, dest='feat_dim', help='feat dimension')
    parser.add_argument('--hidden_dim', type=int, dest='hidden_dim', help='hidden layer dimension')
    parser.add_argument('--output_dim', type=int, dest='output_dim', help='classes of dataset')
    parser.add_argument('--num_layer', type=int, dest='num_layer', help='number of layers')
    parser.add_argument('--seed', type=int, dest='seed', help='seed')
    parser.add_argument('--gpu', type=str, dest='gpu', help='gpu')
    parser.add_argument('--model_name', type=str, dest='model_name', help='model:GCN,GIN,GAT,GraphSAGE')
    parser.add_argument('--max_num_nodes', type=str, dest='max_num_nodes', help='the shape[1] of features matrix')
    parser.add_argument('--batch_size', type=int, dest='batch_size', help='Batch size')
    parser.add_argument('--cuda', dest='cuda', type=int, help="if cuda")
    parser.add_argument('--lr_clf', dest='lr_clf', type=float, help="Clf learning rate")
    parser.add_argument('--epochs', dest='epochs', type=int, help="epochs")
    parser.add_argument('--dropout', dest='dropout', type=float, help="dropout")
    parser.add_argument('--lr_gamma', dest='lr_gamma', type=float, help="learning rate decay rate")
    parser.add_argument('--lr_patience', dest='lr_patience', type=float, help="learning rate decay patience")
    parser.add_argument('--lr_decay', dest='lr_decay', type=int, help="if learning rate decay")
    parser.add_argument('--train_type', dest='train_type', type=str, help="train, retrain")
    parser.add_argument('--split', dest='split', type=str, help='split: scaffold or random')
    parser.add_argument('--ratio', dest='ratio', type=float, help='modified edges ratio')
    parser.add_argument('--iter', dest='iter', type=int, help='aug iterations')
    parser.add_argument('--sample', dest='sample', type=str, help='sample')
    parser.add_argument('--task_flag', dest='task_flag', type=int, help="task_flag=1: multi_task")
    parser.add_argument('--task_num', type=int, dest='task_num', help="task_num")
    parser.set_defaults(task_type='class',
                        datadir='data',
                        bmname='bbbp',
                        task=0,
                        feat_dim=7,
                        hidden_dim=128,
                        num_layer=3,
                        seed=0,
                        gpu='5',
                        output_dim=2,
                        model_name='GCN',
                        max_num_nodes=0,
                        batch_size=128,
                        cuda=1,
                        lr_clf=0.001,
                        epochs=100,
                        dropout=0.5,
                        lr_gamma=0.9,
                        lr_patience=50,
                        lr_decay=0,
                        train_type='train',
                        split='scaffold',
                        ratio=0.15,
                        iter=1,
                        sample='scaffold',
                        task_flag=0,
                        task_num=1)
    return parser.parse_args()

