import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super(Encoder, self).__init__()

        self.input_dim=input_dim
        self.hid_dim=hid_dim
        self.n_layers=n_layers
        self.n_heads=n_heads
        self.pf_dim=pf_dim
        self.encoder_layer=encoder_layer
        self.self_attention=self_attention
        self.positionwise_feedforward=positionwise_feedforward
        self.dropout=dropout
        self.device=device

        self.tok_embedding=nn.Embedding(input_dim, hid_dim) #input_dim 词典大小 ，词嵌入的纬度
        self.pos_embedding=nn.Embedding(1000,hid_dim)

        #构造n_layers个encoder_layer
        self.layers=nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])

        #dropoout 层，输出和输入形状相等
        self.do=nn.Dropout(dropout)

        self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        #src = [batch size, src sent len]
        #src_mask = [batch size, src sent len]
        pos=torch.arange(0,src.shape[1]).unsqueeze(0).repeat(src.shape[0],1).to(self.device)
        #pos构造batchsize行，个【0-sentlen】
        src=self.do((self.tok_embedding(src)*self.scale)+self.pos_embedding(pos))
        #src = [batch size, src sent len, hid dim]

        for layer in self.layers:
            src=layer(src, src_mask)

        return src