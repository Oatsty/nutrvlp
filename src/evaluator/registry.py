from .all import *

def get_evaluator(config, device):
    eval_name = config.EVAL.NAME
    if eval_name == 'nutr_ht_all':
        return NutrHTAllEvaluator(config, device)
    elif eval_name == 'nutr_ht_direct':
        return NutrHTDirectEvaluator(config, device)
    elif eval_name == 'nutr_ht_direct_ingrs':
        return NutrHTDirectIngrsEvaluator(config, device)
    elif eval_name == 'vlp':
        return VLPEvaluator(config, device)
    elif eval_name == 'nutr_vlp_all':
        return NutrVLPAllEvaluator(config, device)
    elif eval_name == 'nutr_vlp_direct':
        return NutrVLPDirectEvaluator(config, device)
    elif eval_name == 'nutr_vlp_direct_ingrs':
        return NutrVLPDirectIngrsEvaluator(config, device)
    elif eval_name ==  'tfood':
        return TFoodEvaluator(config, device)
    elif eval_name ==  'tfood_clip':
        return TFoodClipEvaluator(config, device)
    elif eval_name == 'nutr_tfood_all':
        return NutrTFoodAllEvaluator(config, device)
    elif eval_name == 'nutr_tfood_direct':
        return NutrTFoodDirectEvaluator(config, device)
    elif eval_name == 'nutr_tfood_direct_ingrs':
        return NutrTFoodDirectIngrsEvaluator(config, device)
    elif eval_name == 'old':
        return OldEvaluator(config, device)
    elif eval_name == 'nutr':
        return NutrEvaluator(config, device)
    elif eval_name == 'car_nutr':
        return NutrCarEvaluator(config, device)
    elif eval_name == 'nutr_only_ht':
        return NutrOnlyHTEvaluator(config, device)
    elif eval_name == 'nutr_ingr_only_ht':
        return NutrIngrOnlyHTEvaluator(config, device)
    else:
        ValueError(f'unknown evaluator {eval_name}')
        return NutrHTAllEvaluator(config, device)
