# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_dist(m1: torch.Tensor, m2: torch.Tensor):
    """
    Distance between image and recipe embeddings

    Args:
        im (torch.Tensor): image embeddings with shape (B x D) or (D)
        rec (torch.Tensor): rec embeddings with shape (B x D) or (D)
    """
    return 1 - m1 @ m2.T

class TripletLoss(nn.Module):
    """Triplet loss class

    Args:
        margin (float): loss margin
    """
    def __init__(self, margin=0.3, reduction=None):
        super(TripletLoss, self).__init__()
        self.distance_function = cos_dist
        self.margin = margin
        self.reduction = reduction
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def loss(self, im: torch.Tensor, rec: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        i2r_scores = self.distance_function(F.normalize(im, dim=-1),F.normalize(rec, dim=-1))
        pos = torch.eye(im.size(0),dtype=torch.bool).to(im.device)
        # neg = torch.logical_not(pos).to(im.device)

        # positive similarity
        d1 = i2r_scores.diag().view(im.size(0), 1)
        d2 = d1.T

        y = torch.ones(i2r_scores.size(0)).to(im.device)

        # image anchor to recipe positive
        d1 = d1.expand_as(i2r_scores) # bs x bs
        # recipe anchor to image positive
        d2 = d2.expand_as(i2r_scores) #bs x bs

        y = y.expand_as(i2r_scores)

        # compare every diagonal score to scores in its column
        # recipe retrieval
        cost_im = self.ranking_loss(i2r_scores, d1, y)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_rec = self.ranking_loss(i2r_scores, d2, y)

        # clear diagonals
        cost_rec = cost_rec.masked_fill_(pos, 0)
        cost_im = cost_im.masked_fill_(pos, 0)
        if isinstance(weight,torch.Tensor):
            cost_rec *= weight
            cost_im *= weight
        if self.reduction == 'mean':
            return (cost_rec + cost_im).mean()
        else:
            return (cost_rec + cost_im).sum() / (i2r_scores.shape[0] - 1)

    def forward(self, im: torch.Tensor, rec: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.loss(im,rec)

class MultiTripletLoss(TripletLoss):
    def __init__(self, margin=0.3, reduction=None):
        super(MultiTripletLoss, self).__init__(margin, reduction)
    def forward(self, target_dict: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        keys = list(target_dict.keys())
        losses = torch.tensor(0.,device=target_dict[keys[0]].device)
        count = 0
        for src_key in keys:
            keys.remove(src_key)
            if not len(keys):
                break
            for tgt_key in keys:
                loss = self.loss(target_dict[src_key],target_dict[tgt_key])
                losses += loss
                count += 1
        return losses / count

class WeightedTripletLoss(TripletLoss):
    def __init__(self, margin=0.3, reduction=None):
        super(WeightedTripletLoss, self).__init__(margin, reduction)

    def cal_weight(self, nutrs:torch.Tensor, num_nutrs:int = 4, scale=0.1, **kwargs) -> torch.Tensor:
        sum_weight = torch.zeros((len(nutrs), len(nutrs)), dtype=torch.float, device=nutrs.device)
        for i in range(num_nutrs):
            nutr_combs = torch.combinations(nutrs[:,i],with_replacement=True)
            errs = torch.abs(nutr_combs[:,0] - nutr_combs[:,1])
            weight = torch.zeros((len(nutrs), len(nutrs)), dtype=torch.float, device=nutrs.device)
            triu_indices = torch.triu_indices(len(nutrs), len(nutrs))
            weight[triu_indices[0], triu_indices[1]] = errs
            weight = weight + weight.T
            sum_weight += weight
        return sum_weight * scale

    def forward(self,im: torch.Tensor, rec: torch.Tensor, nutrs: torch.Tensor, **kwargs):
        weight = self.cal_weight(nutrs,**kwargs)
        return self.loss(im,rec,weight)

class WeightedMultiTripletLoss(WeightedTripletLoss):
    def __init__(self, margin=0.3, reduction=None):
        super(WeightedMultiTripletLoss, self).__init__(margin, reduction)
    def forward(self, target_dict: Dict[str, torch.Tensor], nutrs: torch.Tensor, **kwargs) -> torch.Tensor:
        weight = self.cal_weight(nutrs)
        keys = list(target_dict.keys())
        losses = torch.tensor(0.,device=target_dict[keys[0]].device)
        count = 0
        for src_key in keys:
            keys.remove(src_key)
            if not len(keys):
                break
            for tgt_key in keys:
                loss = self.loss(target_dict[src_key],target_dict[tgt_key],weight=weight)
                losses += loss
                count += 1
        return losses / count

def get_loss(config):
    loss_name = config.TRAIN.LOSS
    if loss_name == 'triplet':
        return TripletLoss(), MultiTripletLoss()
    elif loss_name == 'weighted_triplet':
        return WeightedTripletLoss(), WeightedMultiTripletLoss()
    else:
        ValueError(f'unimplemented {loss_name} loss')
        return TripletLoss(), MultiTripletLoss()
