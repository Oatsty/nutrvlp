import sys
import os
sys.path.append('/home/parinayok/food.com_net')
sys.path.append('/home/parinayok/food.com_net/src')
import argparse
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import clip
import transformers

from src.dataset.transforms import Compose, ListDictsToDictLists, PadTensors, StackTensors
from src.model.lora_clip import ClipLoraTextEncoder

class DescriptionEncoder(nn.Module):
    def __init__(self, model_name: str ="meta-llama/Meta-Llama-3-8B-Instruct", device: torch.device = torch.device('cuda'), batch_size = 32, **kwargs) -> None:
        super(DescriptionEncoder, self).__init__()
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
        )
        self.text_encoder = ClipLoraTextEncoder(device, **kwargs)

    def generate_text(self, title: list[str], ingrs: list[list[str]], instrs: list[list[str]]):
        # generated_texts = []
        messages = []
        # prompts = []
        with torch.no_grad():
            for i in range(len(title)):
                message = [
                    {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible and answers are around 30 words"},
                    {"role": "user", "content": f"Give you a food recipe: title:{title[i]}, ingredients: {ingrs[i]} and instructions:{instrs[i]}, return the visual description of the food made according to this recipe. The description should be objective and informative."},
                ]
                messages.append(message)
                # prompt = self.pipeline.tokenizer.apply_chat_template(
                #     messages,
                #     tokenize=False,
                #     add_generation_prompt=True,
                # )
                # prompts.append(prompt)
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id
            outputs = self.pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                batch_size=self.batch_size
            )
            generated_texts = [res[0]['generated_text'][-1]['content'].replace('assistant\n\n','') for res in outputs]
            # generated_texts.append(generated_text)
        return generated_texts

    def tokenize_text(self, texts: list[str]):
        tokenized_texts = []
        for text in texts:
            try:
                tokenized_text = clip.tokenize(text,context_length=300)
            except:
                tokenized_text = clip.tokenize(' '.join(text.split(' ')[:30]),context_length=300)
            tokenized_text = tokenized_text[tokenized_text != 0]
            tokenized_texts.append(tokenized_text)
        return tokenized_texts

    def create_tokenized_tensor(self, tokenized_texts: list[torch.Tensor]):
        max_len = 77
        tokenized_texts_tensor = torch.stack([torch.cat([l,torch.zeros(max_len - len(l),dtype=torch.int32)]) if len(l) < max_len else l[:max_len] for l in tokenized_texts]).to(self.device)
        return tokenized_texts_tensor

    def forward(self, title: list[str], ingrs: list[list[str]], instrs: list[list[str]]):
        generated_texts = self.generate_text(title, ingrs, instrs)
        tokenized_texts = self.tokenize_text(generated_texts)
        tokenized_texts_tensor = self.create_tokenized_tensor(tokenized_texts)
        return generated_texts, tokenized_texts_tensor

class Layer1(Dataset):
    def __init__(self, path_layer1) -> None:
        super().__init__()
        self.path_layer1 = path_layer1
        with open(self.path_layer1) as f:
            layer1_ = json.load(f)
        if isinstance(layer1_, list):
            self.layer1 = {data["id"]: data for data in tqdm(layer1_)}
        else:
            self.layer1 = {key: {**data, 'id': key} for key, data in tqdm(layer1_.items())}
        self.key_list = list(self.layer1.keys())

    def __len__(self):
        return len(self.key_list)

    def get_key(self, index):
        return self.key_list[index]

    def __getitem__(self, index):
        key = self.get_key(index)
        return self.layer1[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-layer", type=str, default='/srv/datasets2/recipe1m+/utfood_3.10/layer1.json', help="path to layer 1")
    parser.add_argument("--output-embeds",type=str,default='/srv/datasets2/recipe1m+/utfood_3.10/clip_text/tokenized_descriptions.pt')
    parser.add_argument("--output-texts",type=str,default='/srv/datasets2/recipe1m+/utfood_3.10/clip_text/descriptions.pt')

    args=parser.parse_args()
    out_texts_path = args.output_texts
    out_embeds_path = args.output_embeds
    path_layer1 = args.path_layer

    os.makedirs(os.path.dirname(out_texts_path),exist_ok=True)
    os.makedirs(os.path.dirname(out_embeds_path),exist_ok=True)

    batch_size = 8

    dataset = Layer1(path_layer1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=Compose([
            ListDictsToDictLists(),
            PadTensors(value=0),
            StackTensors()
        ])
    )

    model = DescriptionEncoder(batch_size=batch_size)

    embeds = {}
    all_texts = {}
    for batch in tqdm(dataloader):
        indices = batch['id']
        titles = batch['title']
        ingrs = batch['ingredients']
        instrs = batch['instructions']
        texts, outs = model(titles,ingrs,instrs)
        outs = outs.cpu()
        for idx, out in zip(indices, outs):
            embeds[idx] = out
        for idx, text in zip(indices, texts):
            all_texts[idx] = text

    with open(out_embeds_path,'wb') as f:
        pickle.dump(embeds,f)
    with open(out_texts_path,'wb') as f:
        pickle.dump(all_texts,f)

if __name__ == '__main__':
    main()
