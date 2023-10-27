import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def eval_rocauc(input_dict):
    '''
        compute ROC-AUC averaged across tasks
    '''
    y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            y = np.array(y_true[:, i], dtype=np.float)
            y[y < 0] = np.nan
            is_labeled = y == y
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}


def eval_rmse(input_dict):
    '''
        compute RMSE score averaged across tasks
    '''
    y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return {'rmse': sum(rmse_list) / len(rmse_list)}