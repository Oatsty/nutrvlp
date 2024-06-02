from .all import *

def get_evaluator(config, device):
    eval_name = config.EVAL.NAME
    if eval_name == 'nutr_ht_all':
        return NutrHTAllEvaluator(config, device)
    elif eval_name == 'nutr_ht_direct':
        return NutrHTDirectEvaluator(config, device)
    elif eval_name == 'nutr_ht_direct_ingrs':
        return NutrHTDirectIngrsEvaluator(config, device)
    elif eval_name == 'nutr_vlp_all':
        return NutrVLPAllEvaluator(config, device)
    elif eval_name == 'nutr_vlp_direct':
        return NutrVLPDirectEvaluator(config, device)
    elif eval_name == 'nutr_vlp_direct_ingrs':
        return NutrVLPDirectIngrsEvaluator(config, device)
    elif eval_name == 'old':
        return OldEvaluator(config, device)
    elif eval_name == 'nutr':
        return NutrEvaluator(config, device)
    else:
        ValueError(f'unknown evaluator {eval_name}')
        return NutrHTAllEvaluator(config, device)
