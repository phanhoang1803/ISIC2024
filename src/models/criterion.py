import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torchvision.ops import focal_loss

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
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
                A weight balancing factor for class 1, 
                default is 0.25 as mentioned in reference Lin et al., 2018. 
                The weight for class 0 is 1.0 - alpha.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
                    
    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
    
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

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
        return FocalLoss(alpha=pos_weight)(outputs, targets)
    elif loss == 'pauc':
        return PAUCLoss()(targets, outputs)
    elif loss == 'mse':
        return nn.MSELoss()(torch.nn.Sigmoid()(outputs), targets)
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


