import torch

def cal_mae(gt: torch.Tensor, pred: torch.Tensor):
    return torch.abs(pred - gt).mean(-1).sum()

def cal_iou(input_1: list, input_2: list):
    input_1_set = set(input_1)
    input_2_set = set(input_2)
    u = input_1_set.union(input_2_set)
    i = input_1_set.intersection(input_2_set)
    return len(i) / len(u)

def cal_weighted_iou(input_1: list, input_2: list, w_1: list, w_2: list):
    all_weight = sum(w_1) + sum(w_2)
    intersect_weight = 0.
    for i, ingr in enumerate(input_1):
        if ingr in input_2:
            j = input_2.index(ingr)
            intersect_weight += w_1[i] + w_2[j]
    return intersect_weight / all_weight
