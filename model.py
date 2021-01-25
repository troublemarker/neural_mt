import argparse
import math

import torch
from torch import nn
from modules.multihead_attn import MultiheadAttention

from options import add_transformer_args
from modules import MultiheadAttention, PositionEmbedding


class TransformerEncoderLayer(nn.Module):

    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.normalize_before = args.encoder_normalize_before
        self.self_attn = MultiheadAttention(args.encoder_embed_dim,
                                            args.encoder_attention_heads,
                                            self_attention=True,
                                            attn_drop=args.attention_dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.self_attn_layernorm = nn.LayerNorm(args.encoder_embed_dim)
        self.final_layernorm = nn.LayerNorm(args.encoder_embed_dim)

        self.fc1 = nn.Linear(args.encoder_embed_dim, args.encoder_ffn_embed_dim)
        self.activate_fn = nn.ReLU()
        self.activate_dropout = nn.Dropout(args.activation_dropout)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, args.encoder_embed_dim)

    def forward(self, x, x, key_padding_mask, decoder_self_atten_mask):

        residual = x
        if self.normalize_before:
            x = self.self_attn_layernorm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual
        if not self.normalize_before:
            x = self.self_attn_layernorm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layernorm(x)
        x = self.fc1(x)
        x = self.activate_fn(x)
        x = self.activate_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layernorm(x)

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = args.decoder_normalize_before

        self.self_attn = MultiheadAttention(args.encoder_embed_dim,
                                            args.encoder_attention_heads,
                                            self_attention=True,
                                            attn_drop=args.attention_dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.self_attn_layernorm = nn.LayerNorm(args.encoder_embed_dim)
        self.final_layernorm = nn.LayerNorm(args.encoder_embed_dim)

        self.fc1 = nn.Linear(args.encoder_embed_dim, args.encoder_ffn_embed_dim)
        self.activate_fn = nn.ReLU()
        self.activate_dropout = nn.Dropout(args.activation_dropout)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, args.encoder_embed_dim)

        self.encoder_attn = MultiheadAttention(args.encoder_embed_dim,
                                            args.encoder_attention_heads,
                                            self_attention=True,
                                            attn_drop=args.attention_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self):
        pass


class TransformerEncoder(nn.Module):

    def __init__(self, args, src_dict, encoder_embed_tokens):
        super(TransformerEncoder, self).__init__()
        self.emb_token = encoder_embed_tokens
        self.position_embed = PositionEmbedding()
        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(TransformerEncoderLayer(args))

        if getattr(args, "layernorm_embedding", False):
            self.embedding_layernorm_ = nn.LayerNorm(self.emb_dim)
        else:
            self.embedding_layernorm = None


    def forward(self, src_tokens):
        x = self.emb_token(src_tokens) * math.sqrt(self.emb_dim)
        x = self.position_embed(x)
        if self.self.embedding_layernorm:
            x = self.self.embedding_layernorm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x


class TransformerDecoder(nn.Module):

    def __init__(self, args, trg_dict, decoder_embed_tokens):
        super(TransformerDecoder, self).__init__()
        # self.padding_idx = padding_idx
        # self.emb_dim = emb_dim

        self.position_embed = PositionEmbedding()
        self.dropout = nn.Dropout()
        self.layers = []
        for i in range(args.decoder_layers):
            self.layers.append(TransformerEncoderLayer())

        if getattr(args, "layernorm_embedding", False):
            self.embedding_layernorm_ = nn.LayerNorm(self.emb_dim)
        else:
            self.embedding_layernorm = None

        if args.share_all_embeddings:
            self.output_projection = nn.Linear(
                decoder_embed_tokens.weight.shape[1],
                decoder_embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(trg_dict), bias=False
            )

    def forward(self, encoder_output, trg_tokens):
        x = self.emb_token(trg_tokens) * math.sqrt(self.emb_dim)
        x = self.position_embed(x)
        x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        test = torch.zeros((3, 4))
        test.eq()
        if trg_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = trg_tokens.eq(self.padding_idx)

        future_mask = self.get_future_mask()

        for layer in self.layers:
            x = layer(x)

    def get_future_mask(self):
        pass



class Transformer(nn.Module):
    def __init__(self, args, src_dict, trg_dict, encoder_embed_tokens, decoder_embed_tokens):
        self.encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        self.decoder = TransformerDecoder(args, trg_dict, decoder_embed_tokens)

    def forward(self, src_token, trg_token):

    def encoder_forward(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_transformer_args(parser)
    args = parser.parse_args()

