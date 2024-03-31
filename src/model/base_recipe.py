import copy
import timm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict


class LearnedPositionalEncoding(nn.Module):
    """ Positional encoding layer

    Parameters
    ----------
    dropout (float): Dropout rate
    num_embeddings (int): Number of embeddings.
    in_dim (int): Embedding dimension
    """

    def __init__(self, dropout=0.1, num_embeddings=50, in_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.weight = Parameter(torch.Tensor(num_embeddings, in_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.in_dim = in_dim
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        seq_len = x.size()[-2]
        if len(x.size()) == 3:
            embeddings = self.weight[:seq_len, :].reshape(1, seq_len, self.in_dim)
        else:
            embeddings = self.weight[:seq_len, :].reshape(1, 1, seq_len, self.in_dim)
        x = x + embeddings
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for recipe embeddings

    Parameters
    ---
    in_dim (int): Input feature dimension.
    num_heads (int): Number of heads.
    num_layers (int): Number of layers.
    """
    def __init__(self, in_dim=300, num_heads=2, num_layers=2):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, ignore_mask=None):
        # batch_first is false for torch.nn transformer encoder
        x = x.permute(1,0,2)
        x = self.encoder(x, src_key_padding_mask=ignore_mask)
        x = x.permute(1,0,2)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for recipe cross attention embeddings

    Parameters
    ---
    dim_in (int): Input feature dimension.
    num_heads (int): Number of heads.
    num_layers (int): Number of layers.
    """
    def __init__(self, dim_in=300, num_heads=2, num_layers=2):
        super(TransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=num_heads)
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, context, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # batch_first is false for torch.nn transformer encoder
        x = x.permute(1, 0, 2)
        context = context.permute(1, 0, 2)
        x = self.encoder(x, context, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        x = x.permute(1, 0, 2)
        return x


class RecipeTransformer(nn.Module):
    """The recipe encoder.

    Parameters
    ---
    vocab_size (int): Recipe vocab size.
    in_dim (int): Output embedding size.
    num_heads (int): Number of heads.
    num_layers (int): Number of layers.
    num_nutrs (int): Number of input nutrition types (4 for cal, fat, carb, protein).

    """
    def __init__(self, vocab_size: int, in_dim: int, num_heads: int, num_layers: int, num_nutrs: int):
        super(RecipeTransformer, self).__init__()

        # model with title, ingredients, instructions, and nutritions as inputs
        names = ['title', 'ingrs', 'instrs', 'nutr']
        # Word embeddings for all vocabs in titles, ingredients, and instructions.
        self.word_embed = nn.Embedding(vocab_size, in_dim)
        self.pos_embed = LearnedPositionalEncoding(in_dim=in_dim)
        # cls tokens for each component when inputting to individual transformer
        self.first_cls_tokens = nn.ParameterDict({
            'title': Parameter(torch.zeros(1, 1, in_dim)),
            'ingrs': Parameter(torch.zeros(1, 1, in_dim)),
            'instrs': Parameter(torch.zeros(1, 1, in_dim)),
        })
        # cls tokens for ingredients and instructions when inputting to sequence merger transformer
        self.second_cls_tokens = nn.ParameterDict({
            'ingrs': Parameter(torch.zeros(1, 1, in_dim)),
            'instrs': Parameter(torch.zeros(1, 1, in_dim)),
        })

        # transformer encoder for individual text
        self.tfs = nn.ModuleDict()
        for name in ['title', 'ingrs', 'instrs']:
            self.tfs[name] = TransformerEncoder(in_dim=in_dim, num_heads=num_heads, num_layers=num_layers)

        # for merging sequence of sequences (ingredients, instructions)
        self.sequence_merger = nn.ModuleDict()
        for name in ['ingrs', 'instrs']:
            self.sequence_merger[name] = TransformerEncoder(in_dim=in_dim, num_heads=num_heads, num_layers=num_layers)

        # # for merging title, ingredients, and instructions together
        # self.all_merger = nn.ModuleDict()
        # for name in ['title', 'ingrs', 'instrs']:
        #     self.all_merger[name] = TransformerDecoder(dim_in=in_dim, num_heads=num_heads, num_layers=num_layers)

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

    def init_weight(self):
        for comp in self.first_cls_tokens.keys():
            nn.init.normal_(self.first_cls_tokens[comp], std=1e-6)

    def forward(self, x: Dict[str,torch.Tensor], nutr: torch.Tensor):
        # word, positional embedding, and cls tokens for each component (title, ingredients, instructions)
        x_1 = x.copy()
        for comp_name in x.keys():
            input = x_1[comp_name]
            input = self.word_embed(input)
            input = self.pos_embed(input)
            if len(x[comp_name].size()) == 2:
                cls_token = self.first_cls_tokens[comp_name].expand(input.shape[0],-1,-1)
            else:
                cls_token = torch.unsqueeze(self.first_cls_tokens[comp_name],0).expand(input.shape[0],input.shape[1],-1,-1)
            input = torch.cat([cls_token,input],-2)
            x_1[comp_name] = input

        out_1 = x_1.copy()
        # individual transformer for each component
        for comp_name in x.keys():
            first_input = copy.copy(x[comp_name])
            input = copy.copy(out_1[comp_name])
            # reshape for ingredients and recipes
            if len(x[comp_name].size()) == 3:
                input = input.reshape(input.size(0)*input.size(1), input.size(2), input.size(3))
                first_input = first_input.reshape(first_input.size(0)*first_input.size(1), first_input.size(2))
            ignore_mask = (first_input == 0)
            ignore_mask = torch.cat([torch.zeros(ignore_mask.shape[0],1,device=ignore_mask.device,dtype=torch.bool),ignore_mask],1)
            out = self.tfs[comp_name](input, ignore_mask)
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
            cls_token = self.second_cls_tokens[comp_name].expand(input.shape[0],-1,-1)
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

class RecipeOnlyTransformer(nn.Module):
    """The recipe only encoder (no nutrients).

    Parameters
    ---
    vocab_size (int): Recipe vocab size.
    in_dim (int): Output embedding size.
    num_heads (int): Number of heads.
    num_layers (int): Number of layers.

    """
    def __init__(self, vocab_size: int, in_dim: int, num_heads: int, num_layers: int):
        super(RecipeOnlyTransformer, self).__init__()

        # model with title, ingredients, instructions as inputs
        names = ['title', 'ingrs', 'instrs']
        # Word embeddings for all vocabs in titles, ingredients, and instructions.
        self.word_embed = nn.Embedding(vocab_size, in_dim)
        self.pos_embed = LearnedPositionalEncoding(in_dim=in_dim)
        # cls tokens for each component when inputting to individual transformer
        self.first_cls_tokens = nn.ParameterDict({
            'title': Parameter(torch.zeros(1, 1, in_dim)),
            'ingrs': Parameter(torch.zeros(1, 1, in_dim)),
            'instrs': Parameter(torch.zeros(1, 1, in_dim)),
        })
        # cls tokens for ingredients and instructions when inputting to sequence merger transformer
        self.second_cls_tokens = nn.ParameterDict({
            'ingrs': Parameter(torch.zeros(1, 1, in_dim)),
            'instrs': Parameter(torch.zeros(1, 1, in_dim)),
        })

        # transformer encoder for individual text
        self.tfs = nn.ModuleDict()
        for name in ['title', 'ingrs', 'instrs']:
            self.tfs[name] = TransformerEncoder(in_dim=in_dim, num_heads=num_heads, num_layers=num_layers)

        # for merging sequence of sequences (ingredients, instructions)
        self.sequence_merger = nn.ModuleDict()
        for name in ['ingrs', 'instrs']:
            self.sequence_merger[name] = TransformerEncoder(in_dim=in_dim, num_heads=num_heads, num_layers=num_layers)

        self.all_merger = nn.Sequential(
            nn.Linear(in_dim*len(names),in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

        self.projectors = nn.ModuleDict()
        for name in ['title','ingrs','instrs']:
            self.projectors[name] = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.ReLU(),
            nn.Linear(in_dim,in_dim)
        )

    def init_weight(self):
        for comp in self.first_cls_tokens.keys():
            nn.init.normal_(self.first_cls_tokens[comp], std=1e-6)

    def forward(self, x: Dict[str,torch.Tensor],**kwargs):
        # word, positional embedding, and cls tokens for each component (title, ingredients, instructions)
        x_1 = x.copy()
        for comp_name in x.keys():
            input = x_1[comp_name]
            input = self.word_embed(input)
            input = self.pos_embed(input)
            if len(x[comp_name].size()) == 2:
                cls_token = self.first_cls_tokens[comp_name].expand(input.shape[0],-1,-1)
            else:
                cls_token = torch.unsqueeze(self.first_cls_tokens[comp_name],0).expand(input.shape[0],input.shape[1],-1,-1)
            input = torch.cat([cls_token,input],-2)
            x_1[comp_name] = input

        out_1 = x_1.copy()
        # individual transformer for each component
        for comp_name in x.keys():
            first_input = copy.copy(x[comp_name])
            input = copy.copy(out_1[comp_name])
            # reshape for ingredients and recipes
            if len(x[comp_name].size()) == 3:
                input = input.reshape(input.size(0)*input.size(1), input.size(2), input.size(3))
                first_input = first_input.reshape(first_input.size(0)*first_input.size(1), first_input.size(2))
            ignore_mask = (first_input == 0)
            ignore_mask = torch.cat([torch.zeros(ignore_mask.shape[0],1,device=ignore_mask.device,dtype=torch.bool),ignore_mask],1)
            out = self.tfs[comp_name](input, ignore_mask)
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
            cls_token = self.second_cls_tokens[comp_name].expand(input.shape[0],-1,-1)
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
        #merge all components (title, ingredients, instructions): to be added

        comp_embs_dict = {name: projector(out_2[name]) for name, projector in self.projectors.items()}
        out_3 = self.all_merger(torch.cat(list(out_2.values()),1))
        return out_3, comp_embs_dict
