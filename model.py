from torch import nn
nn.Module


class MultiheadAttention(nn.Module):



class TransformerLayer(object):

    def __init__(self):
        self.self_attn =

class TransformerEncoder(object):

    def __init__(self, layer_num, token_num, emb_dim):
        self.input_layer = nn.Embedding(token_num, emb_dim)
        self.layers = []
        for i in range(layer_num):
            self.layers.append()


class Transformer(object):
    def __init__(self):
        self.encoder = TransformerEncoder()