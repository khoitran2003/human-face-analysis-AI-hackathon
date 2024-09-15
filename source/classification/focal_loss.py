import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F

def cal_weight(cfg: dict, feature: str):
    df = pd.read_csv(cfg['data']['all_labels'])
    class_weight = len(df[feature])/df[feature].value_counts().sort_index()
    class_weight /= class_weight.sum()
    return(torch.tensor(class_weight.values))


# model_pred size: (bacth_size, num_classes), target_size: (batch_size) 
class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.nllloss = nn.NLLLoss(weight=self.weight, reduce=self.reduction)
    
    def forward(self, predict: torch.tensor, target: torch.tensor):

        log_p = F.log_softmax(predict, dim=-1)
        ce = self.nllloss(log_p, target)

        log_pt = torch.gather(log_p, 1, target.unsqueeze(1))
        pt = log_pt.exp()

        loss = (1 - pt) ** self.gamma * ce # alpha is in ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def focal_loss(feature: str, cfg: dict, pred: torch.tensor,  target: torch.tensor, gamma: int = 2):

    alpha = cal_weight(cfg, feature).to(device=pred.device, dtype=torch.float32)

    focal_loss = Focal_Loss(weight=alpha, gamma=gamma)
    focal_loss = focal_loss(pred, target)
    return focal_loss





        


