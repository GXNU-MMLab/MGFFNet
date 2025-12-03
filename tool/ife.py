import torch
import torch.nn as nn
import torch.nn.functional as F



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class IFE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.5,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)


    def forward(self, tgt, memory):
        batch_size, d_model, height, width = tgt.size()
        tgt = tgt.view(batch_size, d_model, -1).permute(2, 0, 1)
        memory = memory.view(batch_size, d_model, -1).permute(2, 0, 1)
        tgt2, _ = self.multihead_attn(query=tgt, key=memory, value=memory)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt2.permute(1, 2, 0).view(batch_size, d_model, height, width)
        return tgt

