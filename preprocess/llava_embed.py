import sys
import os
sys.path.append('/home/parinayok/food.com_net')
sys.path.append('/home/parinayok/food.com_net/src')
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import re

from transformers import pipeline, LlavaNextProcessor, AutoProcessor, LlavaForConditionalGeneration

import init_config as init_config
from dataset import create_recipe_retrieval_data

class LlavaEncoder(nn.Module):
    def __init__(self, model_name: str ="llava-hf/llava-1.5-7b-hf", device: torch.device = torch.device('cuda'), **kwargs) -> None:
        super(LlavaEncoder, self).__init__()
        self.model_name = model_name
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode(self, image, title: list[str], ingrs: list[list[str]], instrs: list[list[str]]):
        generated_texts = []
        embs = []
        with torch.no_grad():
            for i in range(len(title)):
                # conversation = [
                #     {
                #         "role": "user",
                #         "content": [
                #                 {"type": "text", "text": f"Give you a food recipe: title:{title[i]}, ingredients: {ingrs[i]} and instructions:{instrs[i]}, describe the food made according to the recipe and image. The description should be objective and informative."},
                #                 {"type": "image"},
                #             ],
                #     },
                # ]
                prompt = f"USER: <image>\nGive you a food recipe: title:{title[i]}, ingredients: {ingrs[i]} and instructions:{instrs[i]}, describe the food made according to the recipe and image. The description should be objective and informative. ASSISTANT:"
                # conversation = [
                #     {
                #         "role": "user",
                #         "content": f"Give you a food recipe: title:{title[i]}, ingredients: {ingrs[i]} and instructions:{instrs[i]}, describe the food made according to the recipe and image. The description should be objective and informative.",
                #     },
                # ]
                # prompt = self.processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                # prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                # print(prompt)
                inputs = self.processor(prompt, image[i], return_tensors='pt').to(self.device, torch.float16)
                model_inputs = self.model.prepare_inputs_for_generation(**inputs,  max_new_tokens=200)
                # output_tokens = self.model.generate(**inputs, max_new_tokens=200)
                output = self.model(**model_inputs, output_hidden_states=True)
                # generated_text = self.processor.decode(output_tokens[0], skip_special_tokens=True)
                # generated_text = re.sub(r'.*ASSISTANT:','',generated_text.splitlines()[-1])
                # print(model_inputs['input_ids'].shape)
                emb = output.hidden_states[32][0,-1,:]
                # print(emb.shape)
                # print(output.logits.shape)
            # generated_texts = [res[0]['generated_text'][-1]['content'].replace('assistant\n\n','') for res in outputs]
                # generated_texts.append(generated_text)
                embs.append(emb)
        return generated_texts, embs

    def forward(self, img, title: list[str], ingrs: list[list[str]], instrs: list[list[str]]):
        generated_texts, embs = self.encode(img, title, ingrs, instrs)
        return generated_texts, embs

def main():
    save_dir = '/srv/datasets2/recipe1m+/utfood_3.10/llava/'
    os.makedirs(save_dir,exist_ok=True)
    out_embeds_path = os.path.join(save_dir,'embs.pt')
    out_texts_path = os.path.join(save_dir,'descriptions.pt')
    _, config = init_config.get_arguments()
    config.defrost()
    config.DATA.PATH_DESCRIPTION = None
    config.DATA.PATH_MASK_EMBED = None
    config.DATA.PATH_INGRS = None
    config.DATA.PATH_NUTRS = None
    config.freeze()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LlavaEncoder(device=device)
    model.to(device)
    model.requires_grad_(False)
    phases = ['train', 'val', 'test']
    dataloader = {phase: create_recipe_retrieval_data(config, phase) for phase in phases}
    embeds = {}
    # all_texts = {}
    for phase in ["train", 'val', "test"]:
        for batch in tqdm(dataloader[phase]):
            indices = batch['recipe']['ids']
            pil_img = batch['image']['pil_image']
            titles = batch['recipe']['raw']['title']
            ingrs = batch['recipe']['raw']['ingredients']
            instrs = batch['recipe']['raw']['instructions']
            texts, outs = model(pil_img, titles, ingrs, instrs)
            for idx, out in zip(indices, outs):
                embeds[idx] = out.cpu()
            # for idx, text in zip(indices, texts):
            #     all_texts[idx] = text
        #     break
        # break

    with open(out_embeds_path,'wb') as f:
        pickle.dump(embeds,f)
    # with open(out_texts_path,'wb') as f:
    #     pickle.dump(all_texts,f)

if __name__ == "__main__":
    main()
