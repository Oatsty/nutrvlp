import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, dim_in=300, n_heads=2, n_layers=2):
        super(TransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=n_heads)
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, context, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = x.permute(1, 0, 2)
        context = context.permute(1, 0, 2)
        x = self.encoder(x, context, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        x = x.permute(1, 0, 2)
        return x


class Transformer(nn.Module):
    def __init__(self, dim_in=300, n_heads=2, n_layers=2):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, ignore_mask=None):
        x = self.encoder(x, src_key_padding_mask=ignore_mask)
        return x

class CrossTransformerDecoder(nn.Module):
    def __init__(self, dim_in=300, n_heads=2, n_layers=2, context_1=None, context_2=None, context_3=None):
        super().__init__()
        self.cross_encoders_1 = TransformerDecoder(dim_in=dim_in, n_heads=n_heads, n_layers=n_layers)

        self.cross_encoders_2 = TransformerDecoder(dim_in=dim_in, n_heads=n_heads, n_layers=n_layers)

        self.cross_encoders_3 = TransformerDecoder(dim_in=dim_in, n_heads=n_heads, n_layers=n_layers)

        self.context_1 = context_1 # 1 or 2
        self.context_2 = context_2 # 0 or 2
        self.context_3 = context_3 # 0 or 1

    def forward(self, x1 , x2, x3):# (title, ingrds, instrs)
        cat_dim = 1
        xs = (x1 , x2, x3)
        if self.context_1: #title
            context_1 = xs[self.context_1]
        else:
            context_1 = torch.cat((x2, x3), dim=cat_dim)

        if self.context_2: #ingrds
            context_2 = xs[self.context_2]
        else:
            context_2 = torch.cat((x1, x3), dim=cat_dim)
        if self.context_3: #instrs
            context_3 = xs[self.context_3]
        else:
            context_3 = torch.cat((x1, x2), dim=cat_dim)


        x_1 = self.cross_encoders_1(x1, context=context_1)
        x_2 = self.cross_encoders_2(x2, context=context_2)
        x_3 = self.cross_encoders_3(x3, context=context_3)
        x1 = torch.cat((x_1.mean(1), x_2.mean(1), x_3.mean(1)), dim=1)
        x2 = torch.cat((x_1, x_2, x_3), dim=1)
        return (x1, x2, (x_1, x_2, x_3))



class HTransformerRecipeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim_emb: int, hidden_size: int, num_heads: int, num_layers: int, device):
        super(HTransformerRecipeEmbedding, self).__init__()
        self.device = device
        self.dim_emb = dim_emb

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.n_layers_cross = num_layers
        self.n_heads_cross = num_heads

        # modules
        #ingredients
        self.dim_ingr_in = self.hidden_size
        self.dim_ingr_out = self.dim_ingr_in
        self.encoder_ingrs = Transformer(dim_in=self.dim_ingr_in, n_heads=num_heads, n_layers=num_layers)
        self.encoder_ingr = Transformer(dim_in=self.dim_ingr_in, n_heads=num_heads, n_layers=num_layers)

        #instructions
        self.encoder_instrs = Transformer(dim_in=self.hidden_size, n_heads=num_heads, n_layers=num_layers)
        self.encoder_instr = Transformer(dim_in=self.hidden_size, n_heads=num_heads, n_layers=num_layers)

        #titles
        self.encoder_titles = Transformer(dim_in=self.hidden_size, n_heads=num_heads, n_layers=num_layers)
        context_1 = None
        context_2 = None
        context_3 = None
        self.encoder_cross = CrossTransformerDecoder(dim_in=self.hidden_size, n_heads=self.n_heads_cross,
            n_layers=self.n_layers_cross, context_1=context_1, context_2=context_2, context_3=context_3)

        self.fusion = 'cat'
        self.dim_recipe = self.hidden_size * 3

        self.fc = nn.Linear(self.dim_recipe, self.dim_emb)


    def forward_ingrs_instrs(self, ingrs_out=None, instrs_out=None, titles_out=None):
        fusion_out = self.encoder_cross(x1=titles_out, x2=ingrs_out,x3=instrs_out)
        return fusion_out

    def forward(self, recipe):
        ingrs_out = self.forward_ingrs(recipe['ingrs'])
        instrs_out = self.forward_instrs(recipe['instrs'])
        titles_out = self.forward_titles(recipe['title'])
        x = self.forward_ingrs_instrs(ingrs_out, instrs_out, titles_out)
        return x

    def forward_ingrs(self, ingrs):
        ingrs_ = ingrs.view(ingrs.size(0)*ingrs.size(1), ingrs.size(2))
        emb_out = self.embedding(ingrs_)
        hn = self.encoder_ingr(emb_out).mean(-2)
        hn = hn.view(ingrs.size(0), ingrs.size(1), -1)
        hn = self.encoder_ingrs(hn)
        return hn

    def forward_instrs(self, instrs):
        instrs_ = instrs.view(instrs.size(0)*instrs.size(1), instrs.size(2))
        emb_out = self.embedding(instrs_)
        hn = self.encoder_instr(emb_out).mean(-2)
        hn = hn.view(instrs.size(0), instrs.size(1), -1)
        hn = self.encoder_instrs(hn)
        return hn

    def forward_titles(self, titles):
        emb_out = self.embedding(titles)
        hn = self.encoder_titles(emb_out)
        return hn
