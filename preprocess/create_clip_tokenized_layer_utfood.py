import json
from tqdm import tqdm
import pickle
import argparse
import os

import clip


def create_tokenized_layer(output_path: str, tokenized_text_path: str):
    titles_path = os.path.join(tokenized_text_path, "tokenized_raw_titles.txt")
    with open(titles_path, "rb") as f:
        tokenized_titles = pickle.load(f)

    ingrs_path = os.path.join(tokenized_text_path, "tokenized_raw_ingrs.txt")
    with open(ingrs_path, "rb") as f:
        tokenized_ingrs = pickle.load(f)

    instrs_path = os.path.join(tokenized_text_path, "tokenized_raw_instrs.txt")
    with open(instrs_path, "rb") as f:
        tokenized_instrs = pickle.load(f)

    new_layer = {}
    for i, (k, v) in tqdm(enumerate(tokenized_ingrs.items())):
        new_layer[k] = {}
        new_layer[k]['title'] = tokenized_titles[k].tolist()
        new_layer[k]['ingredients'] = [ingr.tolist() for ingr in tokenized_ingrs[k]]
        new_layer[k]['instructions'] = [instr.tolist() for instr in tokenized_instrs[k]]

    with open(output_path, "w") as f:
        json.dump(new_layer, f)


def tokenize_and_save(entity: str, layer, out_path: str, context_length=32):
    """given a vocab, get and save the indices of each word in the dataset"""

    embedded = {}
    count = 0
    for i, (k, v) in tqdm(enumerate(layer.items())):
        if entity == "title":
            text = v['title'].lower()
            tokenized_text = clip.tokenize(text,context_length=context_length*20)
            tokenized_text = tokenized_text[tokenized_text != 0]
            embedded[k] = tokenized_text

        elif entity == "ingrs":
            tokenized_texts = []
            ingrs = [' '.join(ing.lower().split(' ')[:context_length]) for ing in v["ingredients"]]
            for text in ingrs:
                tokenized_text = clip.tokenize(text,context_length=context_length*20)
                tokenized_text = tokenized_text[tokenized_text != 0]
                tokenized_texts.append(tokenized_text)
            embedded[k] = tokenized_texts

        elif entity == "instrs":
            tokenized_texts = []
            ingrs = [' '.join(ing.lower().split(' ')[:context_length]) for ing in v["instructions"]]
            for text in ingrs:
                tokenized_text = clip.tokenize(text,context_length=context_length*20)
                tokenized_text = tokenized_text[tokenized_text != 0]
                tokenized_texts.append(tokenized_text)
            embedded[k] = tokenized_texts

    print(count, "UNK")
    if out_path is not None:
        with open(out_path, "wb") as f:
            pickle.dump(embedded, f)
    else:
        return embedded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_layer", type=str, default='/srv/datasets2/recipe1m+/utfood_3.10/layer1.json', help="path to layer 1")
    parser.add_argument("--output_path_layer1",type=str,default='/srv/datasets2/recipe1m+/utfood_3.10/clip_text/tokenized_recipe.json')
    parser.add_argument("--output_path_tokenized_texts", type=str, default='/srv/datasets2/recipe1m+/utfood_3.10/clip_text')

    args = parser.parse_args()
    path_layer1 = args.path_layer

    with open(path_layer1, "r") as f:
        layer1 = json.load(f)
    if isinstance(layer1, list):
        layer1_ = {data["id"]: data for data in tqdm(layer1)}
    else:
        layer1_ = layer1

    print("finish reading")

    print("start tokenization...")

    out_path = os.path.join(args.output_path_tokenized_texts, 'tokenized_raw_titles.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embedded = tokenize_and_save(entity='title', layer=layer1_, out_path=out_path)

    out_path = os.path.join(args.output_path_tokenized_texts, 'tokenized_raw_ingrs.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embedded = tokenize_and_save(entity='ingrs', layer=layer1_, out_path=out_path)

    out_path = os.path.join(args.output_path_tokenized_texts, 'tokenized_raw_instrs.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embedded = tokenize_and_save(entity='instrs', layer=layer1_, out_path=out_path)

    print("create and save layer 1...")
    os.makedirs(os.path.dirname(args.output_path_layer1), exist_ok=True)
    create_tokenized_layer(output_path=args.output_path_layer1,tokenized_text_path=args.output_path_tokenized_texts)
