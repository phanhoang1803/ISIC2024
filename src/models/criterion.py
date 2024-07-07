import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def criterion(outputs, targets):
    """
    Calculate the binary cross entropy loss between the model's outputs and the targets.
    
    Args:
        outputs (torch.Tensor): The model's outputs.
        targets (torch.Tensor): The true targets.
    
    Returns:
        torch.Tensor: The binary cross entropy loss.
    """
    # Calculate the binary cross entropy loss using the BCELoss function from torch.nn.
    # The BCELoss function takes the model's outputs and the true targets as input.
    # return nn.BCELoss()(outputs, targets)
    return nn.BCEWithLogitsLoss()(outputs, targets)

def pAUC_score(outputs, targets, min_tpr: float=0.80):
    """
    Calculate the pAUC score based on the model's outputs and targets.
    
    Args:
        outputs (torch.Tensor): The model's outputs.
        targets (torch.Tensor): The true targets.
        min_tpr (float, optional): The minimum true positive rate. Defaults to 0.80.
    
    Returns:
        float: The pAUC score.
    """
    v_gt = abs(np.asarray(targets) - 1)
    v_pred = np.array([1.0 - x for x in outputs])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    
    # Change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

def valid_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80):
    """
    Calculate the valid score based on the solution and submission dataframes, row_id_column_name,
    and minimum true positive rate (tpr).
    
    Args:
        solution (pandas.DataFrame): The dataframe containing the true labels.
        submission (pandas.DataFrame): The dataframe containing the predicted labels.
        row_id_column_name (str): The name of the column containing the row ids.
        min_tpr (float, optional): The minimum true positive rate. Defaults to 0.80.
    
    Returns:
        float: The valid score.
    """
    # Equivalent code that uses sklearn's roc_auc_score
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

