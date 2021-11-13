import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(SelfAttention,self).__init__()

        self.hid_dim=hid_dim
        self.n_heads=n_heads

        assert hid_dim%n_heads==0

        self.w_q=nn.Linear(hid_dim, hid_dim)
        self.w_k=nn.Linear(hid_dim, hid_dim)
        self.w_v=nn.Linear(hid_dim, hid_dim)

        self.fc=nn.Linear(hid_dim,hid_dim)
        self.do=nn.Dropout(dropout)

        self.scale=torch.sqrt(torch.FloatTensor([hid_dim//n_heads])).to(device)
    def forward(self, query, key, value, mask=None):
        bsz=query.shape[0]
        #query = key = value [batch size, sent len, hid dim]

        Q=self.w_q(query)
        K=self.w_k(key)
        V=self.w_v(value)
        #Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        #Q, K, V = [batch size, n heads, sent len, hid dim // n heads]

        # 实现attentionQ*K^T/D
        #Q：[batch size, n heads, sent len, hid dim // n heads]
        #kT:[batch_size,n_heads,hid_dim//n_heads,sent_len]
        energy=torch.matmul(Q,K.permute(0,1,3,2))/self.scale
        #energy = [batch size, n heads, sent len, sent len]

        #当给mask矩阵为0的对应位置替换一个负很大的值后，相应attention的结果就会趋近为0。
        if mask is not None:
            energy=energy.masked_fill(mask==0, -1e10)#masked_fill mask矩阵中index为1的时候 赋value值
        # 实现softmax部分
        attention=self.do(F.softmax(energy, dim=-1))
        #attention = [batch size, n heads, sent len, sent len]

        x=torch.matmul(attention,V)
        #x = [batch size, n heads, sent len, hid dim // n heads]
        x=x.permute(0,2,1,3).contiguous()
        #x = [batch size, sent len, n heads, hid dim // n heads]

        x=x.view(bsz, -1, self.n_heads*(self.hid_dim//self.n_heads))
        #x = [batch size, src sent len, hid dim]

        x=self.fc(x)

        return x