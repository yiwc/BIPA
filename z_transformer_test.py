import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

    def generate_square_subsequent_mask(self, sz):
        # attn_mask:
        # 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        # 3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length, S is the source sequence length.
        # attn_mask ensure that position i is allowed to attend the unmasked positions.
        # If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged.
        # If a BoolTensor is provided, positions with ``True`` is not allowed to attend while ``False`` values will be unchanged.
        # If a FloatTensor is provided, it will be added to the attention weight.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder
        src_mask=~torch.ones_like(src).bool()
        output = self.transformer_encoder(src)
        # output = self.decoder(output)
        return output

if __name__=="__main__":
    pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ntokens = len(vocab.stoi) # the size of vocabulary
    emsize = 2 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(emsize, nhead, nhid, nlayers, dropout).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model.train()
    data=torch.ones([15,15,2]).to(device)
    for i in range(100):
        out=model.forward(data)
        optimizer.zero_grad()
        loss=torch.abs(1.-out).mean()
        loss.backward()
        optimizer.step()
        print(out.mean())

    print(out.shape)