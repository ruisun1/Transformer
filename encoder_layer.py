import torch
import torch.nn as nn
import SelfAttention
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, postionwise_feedforward,dropout,device):
        super(EncoderLayer,self).__init__()

        self.ln=nn.LayerNorm(hid_dim)
        self.sa=self_attention(hid_dim,n_heads,dropout,device)
        self.pf=postionwise_feedforward(hid_dim, pf_dim,dropout)
        self.do=nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src = [batch size, src sent len, hid dim]
        #src_mask = [batch size, src sent len]
        src=self.ln(src+self.do(self.sa(src,src,src,src_mask)))
        src=self.ln(src+self.do(self.pf(src)))

        return src