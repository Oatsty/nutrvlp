import pickle
import json
import numpy as np
import os
import pandas as pd

def cal_iou(input_1, input_2):
    input_1 = set(input_1)
    input_2 = set(input_2)
    u = input_1.union(input_2)
    i = input_1.intersection(input_2)
    return len(i) / len(u)

def cal_mean_iou(out_ingrs,food_ids_per_recipes):
    ious = []
    for recipe_id, pred_ingrs in out_ingrs.items():
        ious.append(cal_iou(food_ids_per_recipes[recipe_id],pred_ingrs))
    iou = sum(ious)/len(ious) * 100
    return iou

def pred_nutr(out_nutr, nutr_per_recipes):
    gts = {
        'energy': [],
        'fat': [],
        'carb': [],
        'protein': [],
    }
    preds = {
        'energy': [],
        'fat': [],
        'carb': [],
        'protein': [],
    }
    absolute_errors = {
        'energy': [],
        'fat': [],
        'carb': [],
        'protein': [],
    }
    percentage_errors = {
        'energy': [],
        'fat': [],
        'carb': [],
        'protein': [],
    }
    symmetric_percentage_errors = {
        'energy': [],
        'fat': [],
        'carb': [],
        'protein': [],
    }

    for recipe_id, pred_nutr in out_nutr.items():
        gt_nutr = nutr_per_recipes[recipe_id]
        for nutr_name in gts.keys():
            gts[nutr_name].append(gt_nutr[nutr_name])
            preds[nutr_name].append(pred_nutr[nutr_name])
            absolute_errors[nutr_name].append(abs(pred_nutr[nutr_name] - gt_nutr[nutr_name]))
            if gt_nutr[nutr_name] < 0.1:
                percentage_errors[nutr_name].append(0)
            else:
                percentage_errors[nutr_name].append(abs(pred_nutr[nutr_name] - gt_nutr[nutr_name])/gt_nutr[nutr_name])
            if gt_nutr[nutr_name] + pred_nutr[nutr_name] < 0.1:
                symmetric_percentage_errors[nutr_name].append(0)
            else:
                symmetric_percentage_errors[nutr_name].append(2*abs(pred_nutr[nutr_name] - gt_nutr[nutr_name])/(gt_nutr[nutr_name] + pred_nutr[nutr_name]))
    return gts, preds, absolute_errors, percentage_errors, symmetric_percentage_errors

def main():
    dirs = [
        # 'deep/direct/base/food_test',
        # 'deep/direct_3_branches/base/food_test',
        # 'deep_ht/direct_ingrs/base/food_test',
        'deep_ht/direct_ingrs_3_branches/base/food_test',
    ]
    output_name = 'all_deep_direct_ingrs_final'
    # dirs = [
    #     'deep_nutr_vlp/base/food_test'
    # ]
    # output_name = 'deep_nutr_vlp'

    with open('/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/food.com_nutr_g_per_recipe.json') as f:
        nutr_g_per_recipes = json.load(f)
    with open('/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/simplified_food_ids_per_recipes.json') as f:
        food_ids_per_recipes = json.load(f)

    index = pd.MultiIndex.from_tuples([
        ('per recipe','mase@1',''),
        ('per recipe','mase@5',''),
        ('per recipe','mase@10',''),
        ('per recipe','mae@1','energy'),
        ('per recipe','mae@1','fat'),
        ('per recipe','mae@1','carb'),
        ('per recipe','mae@1','protein'),
        ('per recipe','mae@5','energy'),
        ('per recipe','mae@5','fat'),
        ('per recipe','mae@5','carb'),
        ('per recipe','mae@5','protein'),
        ('per recipe','mae@10','energy'),
        ('per recipe','mae@10','fat'),
        ('per recipe','mae@10','carb'),
        ('per recipe','mae@10','protein'),
        ('per 100 g','mase@1',''),
        ('per 100 g','mase@5',''),
        ('per 100 g','mase@10',''),
        ('per 100 g','mae@1','energy'),
        ('per 100 g','mae@1','fat'),
        ('per 100 g','mae@1','carb'),
        ('per 100 g','mae@1','protein'),
        ('per 100 g','mae@5','energy'),
        ('per 100 g','mae@5','fat'),
        ('per 100 g','mae@5','carb'),
        ('per 100 g','mae@5','protein'),
        ('per 100 g','mae@10','energy'),
        ('per 100 g','mae@10','fat'),
        ('per 100 g','mae@10','carb'),
        ('per 100 g','mae@10','protein'),
        ('iou','',''),
        ('weighted_iou','',''),
    ])
    output_df = pd.DataFrame([],columns=index)

    for dir2 in dirs:
        output_df.loc[dir2] = pd.Series(dtype='float64')
        dir = dir2
        print(f'analyzing {dir}')
        with open(f'../../out/models/{dir}/out_nutr.json','r') as f:
            out_nutr = json.load(f)
        with open(f'../../out/models/{dir}/out_ingrs.json','r') as f:
            out_ingrs = json.load(f)

        iou = cal_mean_iou(out_ingrs,food_ids_per_recipes)
        output_df.loc[dir2,('iou','','')] = iou
        gts, preds, absolute_errors, percentage_errors, symmetric_percentage_errors = pred_nutr(out_nutr,nutr_g_per_recipes)
        all_mase = []
        for nutr_name in gts.keys():
            mae = sum(absolute_errors[nutr_name])/len(absolute_errors[nutr_name])
            m = np.array(gts[nutr_name]).mean()
            mad = sum(np.abs(np.array(gts[nutr_name]) - m))/(len(gts[nutr_name]))
            mase = mae / mad
            all_mase.append(mase)
            output_df.loc[dir2,('per recipe',f'mae@10',nutr_name)] = mae
        output_df.loc[dir2,('per recipe',f'mase@10','')] = sum(all_mase)/len(list(gts.keys()))

    output_path = f'results_mase/{output_name}.csv'
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    output_df.to_csv(output_path)

if __name__ == '__main__':
    main()
