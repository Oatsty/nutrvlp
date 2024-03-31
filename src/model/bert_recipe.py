import copy
import timm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict, Optional

from transformers import AutoTokenizer, BertModel

from .base_recipe import LearnedPositionalEncoding, TransformerEncoder

class BertRecipeTransformer(nn.Module):
    """The recipe encoder using BERT.

    Parameters
    ---
    in_dim (int): Output embedding size.
    num_heads (int): Number of heads.
    num_layers (int): Number of layers.
    num_nutrs (int): Number of input nutrition types (4 for cal, fat, carb, protein).
    """
    def __init__(self, in_dim: int, num_heads: int, num_layers: int, num_nutrs: int, pretrained: str):
        super(BertRecipeTransformer, self).__init__()
        # model with title, ingredients, instructions, and nutritions as inputs
        names = ['title', 'ingrs', 'instrs', 'nutr']

        # position embeddings and tokenizer
        self.pos_embed = LearnedPositionalEncoding(in_dim=in_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # cls tokens for ingredients and instructions when inputting to sequence merger transformer
        self.cls_tokens = nn.ParameterDict({
            'ingrs': Parameter(torch.zeros(1, 1, in_dim)),
            'instrs': Parameter(torch.zeros(1, 1, in_dim)),
        })

        # BERT model for individual text
        self.first_layer_berts = nn.ModuleDict()
        for name in ['title', 'ingrs', 'instrs']:
            self.first_layer_berts[name] = BertModel.from_pretrained(pretrained) # type: ignore

        # for merging sequence of sequences (ingredients, instructions)
        self.sequence_merger = nn.ModuleDict()
        for name in ['ingrs', 'instrs']:
            self.sequence_merger[name] = TransformerEncoder(in_dim=in_dim, num_heads=num_heads, num_layers=num_layers)

        # nutrition encoder
        self.nutr_encoder = nn.Sequential(
            nn.Linear(num_nutrs,in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

        self.all_merger = nn.Sequential(
            nn.Linear(in_dim*len(names),in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

        self.projectors = nn.ModuleDict()
        for name in ['title','ingrs','instrs','nutr']:
            self.projectors[name] = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

    def forward(self, x: Dict[str,torch.Tensor], nutr: torch.Tensor):
        # add cls token id and eos token id
        x_1 = x.copy()
        for comp_name in x.keys():
            input = x_1[comp_name]
            if len(x[comp_name].size()) == 2:
                cls_token_id = torch.tensor([self.tokenizer.cls_token_id],device=input.device).expand(input.shape[0],-1)
                sep_token_id = torch.tensor([self.tokenizer.sep_token_id],device=input.device).expand(input.shape[0],-1)
            else:
                cls_token_id = torch.tensor([self.tokenizer.cls_token_id],device=input.device).expand(input.shape[0],input.shape[1],-1)
                sep_token_id = torch.tensor([self.tokenizer.sep_token_id],device=input.device).expand(input.shape[0],input.shape[1],-1)
            input = torch.cat([cls_token_id,input,sep_token_id],-1)
            x_1[comp_name] = input

        out_1 = x_1.copy()
        # individual transformer for each component
        for comp_name in x.keys():
            input = copy.copy(out_1[comp_name])
            # reshape for ingredients and recipes
            if len(x[comp_name].size()) == 3:
                input = input.reshape(input.size(0)*input.size(1), input.size(2))
            attention_mask = (input != 0)
            attention_mask[:, 0] = 1
            out = self.first_layer_berts[comp_name](input_ids=input,attention_mask=attention_mask).last_hidden_state
            if len(x[comp_name].size()) == 3:
                out = out.reshape(out_1[comp_name].size(0), out_1[comp_name].size(1), out_1[comp_name].size(2), out.size(-1))
                out = out[:,:,0]
            else:
                out = out[:,0]
            out_1[comp_name] = out

        # position embedding for ingredient and instruction lists
        out_1_w_pos = out_1.copy()
        for comp_name in x.keys():
            if len(x[comp_name].size()) == 2:
                continue
            input = self.pos_embed(out_1_w_pos[comp_name])
            cls_token = self.cls_tokens[comp_name].expand(input.shape[0],-1,-1)
            out_1_w_pos[comp_name] = torch.cat([cls_token,input],-2)

        #  merging sequence of sequences (ingredients, instructions)
        out_2 = out_1_w_pos.copy()
        for comp_name in x.keys():
            if len(x[comp_name].size()) == 2:
                continue
            # first input (only vocab index) for masking
            first_input = copy.copy(x[comp_name])
            attn_mask = first_input > 0
            mask_list = (attn_mask.sum(dim=-1) > 0).bool()
            ignore_mask =  torch.logical_not(mask_list)
            ignore_mask = torch.cat([torch.zeros(ignore_mask.shape[0],1,device=ignore_mask.device,dtype=torch.bool),ignore_mask],1)
            out = self.sequence_merger[comp_name](out_2[comp_name], ignore_mask)
            out = out[:,0]
            out_2[comp_name] = out

        #nutr branch
        nutr_out = self.nutr_encoder(nutr)
        out_2['nutr'] = nutr_out

        #merge all components (title, ingredients, instructions): to be added
        comp_embs_dict = {name: projector(out_2[name]) for name, projector in self.projectors.items()}
        out_3 = self.all_merger(torch.cat(list(out_2.values()),1))
        return out_3, comp_embs_dict

class BertOnlyRecipeTransformer(nn.Module):
    """The recipe encoder using BERT.

    Parameters
    ---
    in_dim (int): Output embedding size.
    num_heads (int): Number of heads.
    num_layers (int): Number of layers.
    num_nutrs (int): Number of input nutrition types (4 for cal, fat, carb, protein).
    """
    def __init__(self, in_dim: int, num_nutrs: int, pretrained: str):
        super(BertOnlyRecipeTransformer, self).__init__()
        # model with title, ingredients, instructions, and nutritions as inputs
        names = ['title', 'ingrs', 'instrs', 'nutr']

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # BERT model for individual text
        self.first_layer_berts = nn.ModuleDict()
        for name in ['title', 'ingrs', 'instrs']:
            self.first_layer_berts[name] = BertModel.from_pretrained(pretrained) # type: ignore

        # nutrition encoder
        self.nutr_encoder = nn.Sequential(
            nn.Linear(num_nutrs,in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

        self.all_merger = nn.Sequential(
            nn.Linear(in_dim*len(names),in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

        self.projectors = nn.ModuleDict()
        for name in ['title','ingrs','instrs','nutr']:
            self.projectors[name] = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

    def forward(self, x: Dict[str,torch.Tensor], nutr: torch.Tensor):
        # add cls token id and eos token id
        x_1 = x.copy()
        for comp_name in x.keys():
            input = x_1[comp_name]
            if len(x[comp_name].size()) == 2:
                cls_token_id = torch.tensor([self.tokenizer.cls_token_id],device=input.device).expand(input.shape[0],-1)
                sep_token_id = torch.tensor([self.tokenizer.sep_token_id],device=input.device).expand(input.shape[0],-1)
            else:
                cls_token_id = torch.tensor([self.tokenizer.cls_token_id],device=input.device).expand(input.shape[0],input.shape[1],-1)
                sep_token_id = torch.tensor([self.tokenizer.sep_token_id],device=input.device).expand(input.shape[0],input.shape[1],-1)
            input = torch.cat([cls_token_id,input,sep_token_id],-1)
            if len(x[comp_name].size()) == 3:
                input = input.reshape(input.shape[0],-1)
            x_1[comp_name] = input

        out_1 = x_1.copy()
        # individual transformer for each component
        for comp_name in x.keys():
            input = copy.copy(out_1[comp_name])
            attention_mask = (input != 0)
            out = self.first_layer_berts[comp_name](input_ids=input,attention_mask=attention_mask).last_hidden_state
            out_1[comp_name] = out[:,0]
        #nutr branch
        nutr_out = self.nutr_encoder(nutr)
        out_1['nutr'] = nutr_out

        #merge all components (title, ingredients, instructions): to be added
        comp_embs_dict = {name: projector(out_1[name]) for name, projector in self.projectors.items()}
        out_fin = self.all_merger(torch.cat(list(out_1.values()),1))
        return out_fin, comp_embs_dict
