import torch
import torch.nn as nn
from decoder import  Decoder
from  encoder import Encoder
from encoder_layer import EncoderLayer
from  decoder_layer import DecoderLayer
from SelfAttention import SelfAttention
from position import PositionwiseFeedforward
from data import valid_iterator
from data import train_iterator
from data import SRC
from data import TRG
from NoamOpt import NoamOpt
import time
import os
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_masks(self, src, trg):

        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]
        #src_mask is if for "pad"
        #tgt mask is for"pad" and "position from 1 to sentlen" (masked multi-attention)
        #trg_pad_mask [batch_size, 1, src_sent_len , 1]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask [batch_size,1,1,src_set_len]
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)#tgt mask is for"pad" and "position from 1 to sentlen"
        #trg_pad_mask [batch_size, 1, src_sent_len , 1]
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))
        print("trg_pad_mask,trg_sub_mask",trg_pad_mask.size(),trg_sub_mask.size())
        trg_mask = trg_pad_mask & trg_sub_mask #broadcasting trg_sub_mask:[senlen,senlen]->[bsz,1,sentlen,sentlen]

        print("trg_mask ",trg_mask.size())

        return src_mask, trg_mask
    def forward(self, src, trg):
        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]

        src_mask, trg_mask = self.make_masks(src, trg)
        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src sent len, hid dim]

        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        #out = [batch size, trg sent len, output dim]

        return out

input_dim=len(SRC.vocab)
hid_dim=512
n_layers=6
n_heads=8
pf_dim=2048
dropout=0.1
enc=Encoder(input_dim,hid_dim,n_layers,n_heads,pf_dim,EncoderLayer,SelfAttention,PositionwiseFeedforward,dropout,device)

output_dim=len(TRG.vocab)
hid_dim=512
n_layers=6
n_heads=8
pf_dim=2048
dropout=0.1
dec=Decoder(output_dim,hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx=SRC.vocab.stoi['<pad>']
model=Seq2Seq(enc,dec,pad_idx,device).to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optimizer = NoamOpt(hid_dim, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg
        src=src.permute(1,0)
        trg=trg.permute(1,0)


        optimizer.optimizer.zero_grad()

        output = model(src, trg[:,:-1])

        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg
            src=src.permute(1,0)
            trg=trg.permute(1,0)

            output = model(src, trg[:,:-1])

            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)

            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 5
CLIP = 1
SAVE_DIR = './models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')