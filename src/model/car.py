from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

import clip
import transformers

from model.sam import SAM, SAM2
from .lora_clip import ClipImageEncoder, ClipLoraImageEncoder, ClipLoraTextEncoder
from .ht import SingleTransformerEncoder, AvgPoolSequence
import logging

logger = logging.getLogger()

class RecipeEncoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, n_heads: int, n_layers: int, device: torch.device, **kwargs) -> None:
        super(RecipeEncoder, self).__init__()
        self.device=device
        # first independent clip encoder for title, ingredient, and instruction
        self.tfs  = nn.ModuleDict()
        for name in ['title', 'ingredients', 'instructions']:
            self.tfs[name] = ClipLoraTextEncoder(device, **kwargs)

        # second transformer for ingredient list and instruction list
        self.merger = nn.ModuleDict()
        for name in ['ingredients', 'instructions']:
            self.merger[name] = SingleTransformerEncoder(dim=hidden_size, n_heads=n_heads, n_layers=n_layers)

        self.merger_recipe = nn.Linear(hidden_size*(3), output_size)

    def preprocss(self, input: torch.Tensor):
        max_len = 77
        if len(input.size()) == 2:
            if input.shape[1] < max_len:
                return torch.cat([input,torch.zeros(input.shape[0],max_len - input.shape[1],dtype=torch.int32).to(self.device)],dim=1)
            else:
                return input[:,:max_len]
        else:
            if input.shape[2] < max_len:
                return torch.cat([input,torch.zeros(input.shape[0],input.shape[1],max_len - input.shape[2],dtype=torch.int32).to(self.device)],dim=2)
            else:
                return input[:,:,:max_len]

    def forward_feature(self, input, name=None):
        '''
        Extracts features for an input using the corresponding encoder (by name)
        '''
        input = self.preprocss(input)
        # check if input is a sequence or a sequence of sequences
        if len(input.size()) == 2:
            # if it is a sequence, the output of a single transformer is used
            ignore_mask = (input == 0)
            # logging.info(input)
            # logging.info(input.shape)
            out = self.tfs[name](input)
            out = AvgPoolSequence(torch.logical_not(ignore_mask), out)
        else:
            # if it's a sequence of sequences, the first encoder is applied
            # to each sentence, and the second on

            # reshape from BxNxTxD to BNxTxD
            input_rs = input.view(input.size(0)*input.size(1), input.size(2))
            ignore_mask = (input_rs == 0)

            # trick to avoid nan behavior with fully padded sentences
            # (due to batching)
            ignore_mask[:, 0] = 0
            out = self.tfs[name](input_rs)
            out = AvgPoolSequence(torch.logical_not(ignore_mask), out)

            # reshape back
            out = out.view(input.size(0), input.size(1), out.size(-1))

            # create mask for second transformer
            attn_mask = input > 0
            mask_list = (attn_mask.sum(dim=-1) > 0).bool()
            out = self.merger[name](out, torch.logical_not(mask_list))
        return out

    def forward(self, title: torch.Tensor, ingrs: torch.Tensor, instrs: torch.Tensor):
        text_features = []
        elems = {'title': title, 'ingredients': ingrs, 'instructions': instrs}
        names = list(elems.keys())
        for name in names:
            # for each recipe component, extracts features and projects them to all other spaces
            input_source = elems[name]
            text_feature = self.forward_feature(input_source, name)
            text_features.append(text_feature)
        recipe_feat = self.merger_recipe(torch.cat(text_features, dim=1))
        recipe_feat = nn.Tanh()(recipe_feat)
        return recipe_feat


class SegmentEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device, score_threshold = 0.95, area_threshold=0.005) -> None:
        super(SegmentEncoder, self).__init__()
        self.device = device
        self.sam_model = SAM(model_name, device)
        self.sam_model.requires_grad_(False)
        self.score_threshold = score_threshold
        self.area_threshold = area_threshold
        self.img_encoder = ClipImageEncoder(device)
        # scale_size = 256
        # crop_size = 224
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # self.transform = transforms.Compose([
        #     # transforms.Resize(scale_size),
        #     transforms.RandomCrop(crop_size),
        #     # transforms.CenterCrop(size),
        #     # transforms.ToTensor(),  # divide by 255 automatically
        #     transforms.Normalize(mean=mean, std=std)
        # ])

    def filter_one(self, outputs):
        filtered_masks = []
        for mask, score in zip(outputs['masks'],outputs['scores']):
            if score < self.score_threshold:
                continue
            area_ratio = mask.sum().item() / (mask.shape[0] * mask.shape[1])
            if area_ratio < self.area_threshold:
                continue
            filtered_masks.append(torch.tensor(mask))
        filtered_masks = torch.stack(filtered_masks).to(self.device)
        return filtered_masks

    def filter(self, outputs):
        all_filtered_masks = []
        n_masks = []
        for output in outputs:
            filtered_masks = self.filter_one(output)
            n_masks.append(len(filtered_masks))
            all_filtered_masks.append(filtered_masks)
        return all_filtered_masks, n_masks

    def no_filter(self, outputs):
        all_masks = []
        n_masks = []
        for output in outputs:
            masks = torch.stack([torch.tensor(m) for m in output['masks']]).to(self.device)
            n_masks.append(len(masks))
            all_masks.append(masks)
        return all_masks, n_masks

    def forward_feature(self, img, img_tensor):
        with torch.no_grad():
            outputs = self.sam_model(img)
        B = len(img_tensor)
        #get mask by filtered with area and score thresholds
        # masks, n_masks = self.filter(outputs)
        masks, n_masks = self.no_filter(outputs)
        all_img_segments = []
        for i in range(B):
            # img_segments = torch.einsum('chw, mhw -> mchw', TF.to_tensor(img[i]), masks[i])
            # img_segment_tensors = self.transform(img_segments)
            img_segment_tensors = torch.einsum('chw, mhw -> mchw', img_tensor[i], masks[i])
            all_img_segments.append(img_segment_tensors)

        # encoder with clip
        img_segments_rs = torch.cat(all_img_segments,dim=0)
        with torch.no_grad():
            img_embs_rs = self.img_encoder(img_segments_rs)

        # reshape back and find avg
        current_index = 0
        all_img_embs = []
        for n_mask in n_masks:
            next_index = current_index + n_mask
            img_emb = img_embs_rs[current_index: next_index].mean(0)
            all_img_embs.append(img_emb)
            current_index = next_index

        img_embs = torch.stack(all_img_embs)
        return img_embs, masks

    def forward(self, img, img_tensor):
        emb, _ = self.forward_feature(img, img_tensor)
        return emb

class DescriptionEncoder(nn.Module):
    def __init__(self, model_name: str ="meta-llama/Meta-Llama-3-8B-Instruct", device: torch.device = torch.device('cuda'), **kwargs) -> None:
        super(DescriptionEncoder, self).__init__()
        # self.model_name = model_name
        self.device = device
        # self.pipeline = transformers.pipeline(
            # "text-generation",
            # model=model_name,
            # model_kwargs={"torch_dtype": torch.bfloat16},
            # device_map=device,
        # )
        self.text_encoder = ClipLoraTextEncoder(device, **kwargs)

    def generate_text(self, title: list[str], ingrs: list[list[str]], instrs: list[list[str]]):
        generated_texts = []
        with torch.no_grad():
            for i in range(len(title)):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible and answers are around 30 words"},
                    {"role": "user", "content": f"Give you a food recipe: title:{title[i]}, ingredients: {ingrs[i]} and instructions:{instrs[i]}, return the visual description of the food made according to this recipe. The description should be objective and informative."},
                ]
                terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                generated_text = outputs[0]['generated_text'][-1]['content']
                generated_texts.append(generated_text)
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
        # max_len = max([len(l) for l in tokenized_texts])
        max_len = 77
        tokenized_texts_tensor = torch.stack([torch.cat([l,torch.zeros(max_len - len(l),dtype=torch.int32)]) if len(l) < max_len else l[:max_len] for l in tokenized_texts]).to(self.device)
        return tokenized_texts_tensor

    def forward(self, tokenized_texts_tensor: torch.Tensor):
        # generated_texts = self.generate_text(title, ingrs, instrs)
        # tokenized_texts = self.tokenize_text(generated_texts)
        # tokenized_texts_tensor = self.create_tokenized_tensor(tokenized_texts)
        # print(tokenized_texts_tensor.shape)
        ignore_mask = (tokenized_texts_tensor == 0)
        out = self.text_encoder(tokenized_texts_tensor)
        out = AvgPoolSequence(torch.logical_not(ignore_mask), out)
        out = nn.Tanh()(out)
        return out

class Car(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, n_heads: int, n_layers: int, segment_model_name: str, llm_model_name: str, device: torch.device, score_threshold = 0.95, area_threshold=0.005, **kwargs) -> None:
        super(Car, self).__init__()
        self.recipe_encoder = RecipeEncoder(output_size, hidden_size, n_heads, n_layers, device, **kwargs)
        # self.segment_encoder = SegmentEncoder(segment_model_name, device, score_threshold, area_threshold)
        self.description_encoder = DescriptionEncoder(llm_model_name, device, **kwargs)
        self.image_encoder = ClipLoraImageEncoder(device,**kwargs)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward_features(self, img: torch.Tensor, pil_img, title: torch.Tensor, ingrs: torch.Tensor, instrs: torch.Tensor, description: torch.Tensor, title_text: Optional[list[str]] = None, ingrs_text: Optional[list[list[str]]] = None, instrs_text: Optional[list[list[str]]] = None):
        embs = {}
        embs['img'] = self.image_encoder(self.normalize(img))
        embs['recipe'] = self.recipe_encoder(title, ingrs, instrs)
        # embs['segment'] = self.segment_encoder(pil_img, img)
        embs['description'] = self.description_encoder(description)
        return embs

    def forward(self, img: torch.Tensor, pil_img, title: torch.Tensor, ingrs: torch.Tensor, instrs: torch.Tensor, description: torch.Tensor, title_text: Optional[list[str]] = None, ingrs_text: Optional[list[list[str]]] = None, instrs_text: Optional[list[list[str]]] = None):
        embs = self.forward_features(img,pil_img,title,ingrs,instrs,description)
        return embs

def create_car_model(config, device):
    model_name = config.MODEL.SEGMENT.NAME
    llm_model_name = config.MODEL.DESCRIPTION.NAME
    output_size = config.MODEL.EMB_DIM
    hidden_size = config.MODEL.RECIPE.HIDDEN_DIM
    n_heads = config.MODEL.RECIPE.NUM_HEADS
    n_layers = config.MODEL.RECIPE.NUM_LAYERS
    return Car(output_size, hidden_size, n_heads, n_layers, model_name, llm_model_name, device).to(device)
