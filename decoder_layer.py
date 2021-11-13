import torch
import torch.nn as nn
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):

        #trg = [batch size, trg sent len, hid dim]
        #src = [batch size, src sent len, hid dim]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask))) #query from decoder input; key and value from encoder output

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg