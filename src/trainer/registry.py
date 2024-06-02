from .nutr import NutrTrainer
from .old import OldTrainer
from .vlp_all import VLPTrainer, VLPNoNutrTrainer, VLP3BranchesTrainer, VLPDirectTrainer, VLPDirect3BranchesTrainer, VLPIngrsOnly3BranchesTrainer, VLPDirectIngrsTrainer, VLPDirectIngrs3BranchesTrainer, VLPDirectIngrsNoNutrTrainer
from .ht_all import HTTrainer, HTNoNutrTrainer, HT3BranchesTrainer, HTDirectTrainer, HTDirect3BranchesTrainer, HTIngrsOnly3BranchesTrainer, HTIngrsOnlyNoNutrTrainer, HTDirectIngrsTrainer, HTDirectIngrs3BranchesTrainer, HTDirectIngrsNoNutrTrainer

def get_trainer(config, device):
    trainer_name = config.TRAIN.NAME
    if trainer_name == 'old':
        return OldTrainer(config, device)
    elif trainer_name == 'nutr':
        return NutrTrainer(config, device)
    elif trainer_name == 'vlp':
        return VLPTrainer(config, device)
    elif trainer_name == 'vlp_no_nutr':
        return VLPNoNutrTrainer(config, device)
    elif trainer_name == 'vlp_3_branches':
        return VLP3BranchesTrainer(config, device)
    elif trainer_name == 'vlp_direct':
        return VLPDirectTrainer(config, device)
    elif trainer_name == 'vlp_direct_3_branches':
        return VLPDirect3BranchesTrainer(config, device)
    elif trainer_name == 'vlp_ingrs_only_3_branches':
        return VLPIngrsOnly3BranchesTrainer(config, device)
    elif trainer_name == 'vlp_direct_ingrs':
        return VLPDirectIngrsTrainer(config, device)
    elif trainer_name == 'vlp_direct_ingrs_3_branches':
        return VLPDirectIngrs3BranchesTrainer(config, device)
    elif trainer_name == 'vlp_direct_ingrs_no_nutr':
        return VLPDirectIngrsNoNutrTrainer(config, device)
    elif trainer_name == 'ht':
        return HTTrainer(config, device)
    elif trainer_name == 'ht_no_nutr':
        return HTNoNutrTrainer(config, device)
    elif trainer_name == 'ht_3_branches':
        return HT3BranchesTrainer(config, device)
    elif trainer_name == 'ht_direct':
        return HTDirectTrainer(config, device)
    elif trainer_name == 'ht_direct_3_branches':
        return HTDirect3BranchesTrainer(config, device)
    elif trainer_name == 'ht_ingrs_only_3_branches':
        return HTIngrsOnly3BranchesTrainer(config, device)
    elif trainer_name == 'ht_ingrs_only_no_nutr':
        return HTIngrsOnlyNoNutrTrainer(config, device)
    elif trainer_name == 'ht_direct_ingrs':
        return HTDirectIngrsTrainer(config, device)
    elif trainer_name == 'ht_direct_ingrs_3_branches':
        return HTDirectIngrs3BranchesTrainer(config, device)
    elif trainer_name == 'ht_direct_ingrs_no_nutr':
        return HTDirectIngrsNoNutrTrainer(config, device)
    else:
        ValueError(f'unknown trainer {trainer_name}')
        return HTTrainer(config, device)
