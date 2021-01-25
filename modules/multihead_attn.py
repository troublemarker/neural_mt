import torch
from torch import nn


class MultiheadAttention(nn.Module):

    def __init__(self, dim, heads, self_attention, attn_drop):

        super(MultiheadAttention, self).__init__()
        self.self_attention = self_attention
        self.dim = float(dim)
        self.heads = heads
        self.head_dim = dim // heads
        self.query_op = nn.Linear(dim, dim)
        self.key_op = nn.Linear(dim, dim)
        self.value_op = nn.Linear(dim, dim)
        self.attn_drop_op = nn.Dropout(attn_drop)

        self.out_op = nn.Linear(dim, dim)

    def forward(self, query, key, key_padding_mask, decoder_self_atten_mask):
        """

        :param x: shape (T, B, C)
        """

        query_len, batch_size, channel = query.shape
        key_len = key.shape[0]

        q = self.query_op(query)

        if self.self_attention:
            value = key = query
        else:
            value = key

        k = self.key_op(key)
        v = self.value_op(value)

        q_split = q.contiguous().view(query_len, batch_size * self.heads, self.head_dim).transpose(0, 1)
        k_split = k.contiguous().view(key_len, batch_size * self.heads, self.head_dim).transpose(0, 1)
        v_split = v.contiguous().view(key_len, batch_size * self.heads, self.head_dim).transpose(0, 1)

        # batch_size * self.heads query_len key_len
        logits = q_split.matmul(k_split.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dim))

        if key_padding_mask:
            logits.masked_fill(key_padding_mask.unsqueeze(1), value=float("-inf"))

        # right-upper is the mask, i.e. float("-inf")
        if decoder_self_atten_mask:
            logits = logits + decoder_self_atten_mask.unsqueeze(0)

        probs = torch.softmax(logits, dim=-1)
        probs = self.attn_drop_op(probs)
        output = probs.matmul(v_split)
        output = output.transpose(0, 1).contiguous().view(query_len, batch_size, self.heads * self.head_dim)

        out = self.out_op(output)
        return out


if __name__ == '__main__':
    mha = MultiheadAttention(20, 4, True, 0.1)
    x = torch.ones((2, 3, 20))
    out = mha(x, x)
    print(out.shape)