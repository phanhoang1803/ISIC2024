import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

class PAUCLoss(nn.Module):
    def __init__(self, min_tpr=0.8):
        super(PAUCLoss, self).__init__()
        self.min_tpr = min_tpr

    def forward(self, y_true, y_pred):
        # Get indices of positive and negative samples
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        n_pos = pos_scores.size(0)
        n_neg = neg_scores.size(0)

        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.0, requires_grad=True)

        pos_scores = pos_scores.view(-1, 1)
        neg_scores = neg_scores.view(1, -1)

        # Calculate the threshold for the minimum TPR
        threshold = torch.quantile(pos_scores, self.min_tpr)
        filtered_pos_scores = pos_scores[pos_scores >= threshold]

        if filtered_pos_scores.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Pairwise differences
        diff = neg_scores - filtered_pos_scores + 1  # Add margin for better separation
        loss = torch.mean(torch.clamp(diff, min=0))

        return loss

class FocalLoss(nn.Module):
    # def __init__(self, alpha=torch.tensor([0.05, 0.95]), gamma=2, reduction='mean'):
    #     super(FocalLoss, self).__init__()
    #     self.alpha = alpha
    #     self.gamma = gamma
    #     self.reduction = reduction

    # def forward(self, inputs, targets):
    #     BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    #     pt = torch.exp(-BCE_loss)
    #     F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

    #     if self.reduction == 'mean':
    #         return torch.mean(F_loss)
    #     elif self.reduction == 'sum':
    #         return torch.sum(F_loss)
    #     else:
    #         return F_loss
    
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int, list)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
        targets = targets.view(-1, 1)

        logpt = -torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(logpt)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

def criterion(outputs, targets, pos_weight=20.0, loss='bce_with_logits'):
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
    if loss == 'bce':
        return nn.BCELoss()(torch.nn.Sigmoid()(outputs), targets)
    elif loss == 'bce_with_logits':
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(outputs.device))(outputs, targets)
        else:
            return nn.BCEWithLogitsLoss()(outputs, targets)
    elif loss == 'focal':
        return FocalLoss()(outputs, targets)
    elif loss == 'pauc':
        return PAUCLoss()(targets, outputs)
    else:
        raise ValueError(f"Invalid loss function: {loss}")

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
    try:
        partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    except ValueError:
        print("ValueError: ROC AUC score is not defined for empty label set.")
        print("raw v_pred:", v_pred)
        v_pred = np.nan_to_num(v_pred)
        print("v_pred processed:", v_pred)
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


