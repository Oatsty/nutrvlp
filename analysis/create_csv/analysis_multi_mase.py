import pickle
import json
import numpy as np
import os
import pandas as pd

def save_im2recipe_dict(im2recipe,saved_ids,dir):
    im2recipe_dict = {}
    for i, sorted_recipe in enumerate(im2recipe):
        avail_sorted_recipe = [saved_ids[idx] for idx in sorted_recipe if idx != i][:20]
        im2recipe_dict[saved_ids[i]] = avail_sorted_recipe
    save_path = f'../results/{dir}.json'
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(f'../results/{dir}.json','w') as f:
        json.dump(im2recipe_dict,f,indent=2)
    return im2recipe_dict

def cal_iou(input_1, input_2):
    input_1 = set(input_1)
    input_2 = set(input_2)
    u = input_1.union(input_2)
    i = input_1.intersection(input_2)
    return len(i) / len(u)
def cal_weighted_iou(input_1, input_2, w_1, w_2):
    all_weight = sum(w_1) + sum(w_2)
    intersect_weight = 0.
    for i, ingr in enumerate(input_1):
        if ingr in input_2:
            j = input_2.index(ingr)
            intersect_weight += w_1[i] + w_2[j]
    return intersect_weight / all_weight

def cal_mean_iou(im2recipe_dict,food_ids_per_recipes,weighted_food_ids_per_recipes):
    ious = []
    weighted_ious = []
    for recipe_id, results in im2recipe_dict.items():
        result = results[0]
        ious.append(cal_iou(food_ids_per_recipes[recipe_id],food_ids_per_recipes[result]))
        weighted_ious.append(cal_weighted_iou(food_ids_per_recipes[recipe_id],food_ids_per_recipes[result],weighted_food_ids_per_recipes[recipe_id]['weight'],weighted_food_ids_per_recipes[result]['weight']))
    iou = sum(ious)/len(ious) * 100
    weighted_iou = sum(weighted_ious)/len(weighted_ious) * 100
    return iou, weighted_iou

def pred_nutr(im2recipe_dict, nutr_per_recipes, top_k: int):

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

    for recipe_id, results in im2recipe_dict.items():
        top_k_results = results[:top_k]
        gt_nutr = nutr_per_recipes[recipe_id]
        pred_nutr = {}
        for nutr_name in gts.keys():
            pred_nutr[nutr_name] = sum([nutr_per_recipes[result][nutr_name] for result in top_k_results]) / top_k
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
        'vlpcook/base',
        'clip/img/test',
        'clip/text/test',
        'clip/text_img/test',
        'nutr_vlp/base/food_test',
        'deep/direct_ingrs_3_branches/base/food_test_recipe_img'
    ]
    output_name = 'new_all_comaparison'

    # dirs = [
    #     'deep/base/base/food_test_all_domains',
    #     'deep/base_3_branches/base/food_test_all_domains',
    #     'deep/direct/base/food_test_all_domains',
    #     'deep/direct_3_branches/base/food_test_all_domains',
    #     'deep/direct_ingrs/base/food_test_all_domains',
    #     'deep/direct_ingrs_3_branches/base/food_test_all_domains',
    # ]
    # output_name = 'comparison_deep_all_domains'

    # dirs = [
    #     'deep/base/base/food_test_recipe_img',
    #     'deep/base_3_branches/base/food_test_recipe_img',
    #     'deep/direct/base/food_test_recipe_img',
    #     'deep/direct_3_branches/base/food_test_recipe_img',
    #     'deep/direct_ingrs/base/food_test_recipe_img',
    #     'deep/direct_ingrs_3_branches/base/food_test_recipe_img',
    # ]
    # output_name = 'comparison_deep_recipe_img'

    # dirs = [
    #     'deep/direct_ingrs_3_branches/no_nutr/food_test_recipe_img',
    #     'deep/direct_ingrs_3_branches/triplet/food_test_recipe_img',
    #     'deep/base_3_branches/base/food_test_recipe_img',
    #     'deep/direct_ingrs/base/food_test_recipe_img',
    #     'deep/direct_ingrs_3_branches/ingrs_only/food_test_recipe_img',
    #     'deep/direct_ingrs_3_branches/base/food_test_recipe_img',
    # ]

    # output_name = 'ablation_deep_recipe_img'

    # dirs = [
    #     'deep/base/vlpcook/food_test_recipe_img',
    #     'deep/base/triplet/food_test_recipe_img',
    #     'deep/base/base/food_test_recipe_img',
    #     'deep/base_3_branches/base/food_test_recipe_img',
    #     'deep/direct_ingrs_3_branches/ingrs_only/food_test_recipe_img',
    #     'deep/direct_ingrs_3_branches/base/food_test_recipe_img',
    # ]

    # output_name = 'complete_ablation_deep_recipe_img'

    with open('/home/parinayok/nutr1m/data_crawl/food.com_annotated_nutr_per_recipe_old.json') as f:
        nutr_per_recipes = json.load(f)
    with open('/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/food.com_nutr_g_per_recipe.json') as f:
        nutr_g_per_recipes = json.load(f)
    with open('/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/simplified_food_ids_per_recipes.json') as f:
        food_ids_per_recipes = json.load(f)
    with open('/srv/datasets2/recipe1m+/food.com_data_rescaled/nutr/simplified_weighted_food_ids_per_recipes.json') as f:
        weighted_food_ids_per_recipes = json.load(f)

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

    for dir in dirs:
        dir2 = ' '.join(dir.split('/')[:-1])
        output_df.loc[dir2] = pd.Series(dtype='float64')
        print(f'analyzing {dir}')
        im2recipe = np.load(f'../../out/models/{dir}/im2recipe.npy')
        with open(f'../../out/models/{dir}/saved_ids','rb') as f:
            saved_ids = pickle.load(f)

        im2recipe_dict = save_im2recipe_dict(im2recipe,saved_ids,dir)
        iou, weighted_iou = cal_mean_iou(im2recipe_dict,food_ids_per_recipes,weighted_food_ids_per_recipes)
        output_df.loc[dir2,('iou','','')] = iou
        output_df.loc[dir2,('weighted_iou','','')] = weighted_iou
        for top_k in [1,5,10]:
            #topk per recipe
            gts, preds, absolute_errors, percentage_errors, symmetric_percentage_errors = pred_nutr(im2recipe_dict,nutr_per_recipes,top_k)
            all_mase = []
            for nutr_name in gts.keys():
                mae = sum(absolute_errors[nutr_name])/len(absolute_errors[nutr_name])
                m = np.array(gts[nutr_name]).mean()
                mad = sum(np.abs(np.array(gts[nutr_name]) - m))/(len(gts[nutr_name]))
                mase = mae / mad
                all_mase.append(mase)
                output_df.loc[dir2,('per recipe',f'mae@{top_k}',nutr_name)] = mae
            output_df.loc[dir2,('per recipe',f'mase@{top_k}','')] = sum(all_mase)/len(list(gts.keys()))

            #topk per 100 g
            gts, preds, absolute_errors, percentage_errors, symmetric_percentage_errors = pred_nutr(im2recipe_dict,nutr_g_per_recipes,top_k)
            all_mase = []
            for nutr_name in gts.keys():
                mae = sum(absolute_errors[nutr_name])/len(absolute_errors[nutr_name])
                m = np.array(gts[nutr_name]).mean()
                mad = sum(np.abs(np.array(gts[nutr_name]) - m))/(len(gts[nutr_name]))
                mase = mae / mad
                all_mase.append(mase)
                output_df.loc[dir2,('per 100 g',f'mae@{top_k}',nutr_name)] = mae
            output_df.loc[dir2,('per 100 g',f'mase@{top_k}','')] = sum(all_mase)/len(list(gts.keys()))

    output_path = f'results_mase/{output_name}.csv'
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    output_df.to_csv(output_path)

if __name__ == '__main__':
    main()
