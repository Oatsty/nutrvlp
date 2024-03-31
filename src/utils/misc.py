
def convert_state_dict(org_state_dict):
    all_model_state_dict = {}
    for k, v in org_state_dict['network'].items():
        if 'image_embedding.convnet.clip_model' in k:
            new_k = k.replace('image_embedding.convnet.clip_model','image_embedder.encoder')
            all_model_state_dict[new_k] = v
            continue
        if 'image_embedding.' in k:
            new_k = k.replace('image_embedding.','image_embedder.')
            all_model_state_dict[new_k] = v
            continue
        if 'recipe_embedding.' in k:
            new_k = k.replace('recipe_embedding.','recipe_embedder.')
            all_model_state_dict[new_k] = v
        if 'proj_recipe.' in k:
            all_model_state_dict[k] = v
        if 'proj_image.' in k:
            all_model_state_dict[k] = v
    return all_model_state_dict
