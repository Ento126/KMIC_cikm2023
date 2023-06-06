import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

class Context(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers):
        super(Context, self).__init__()
        self.dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.GRU(input_size=self.dim,
                        hidden_size=self.hidden_dim,
                        num_layers=self.n_layers,
                        batch_first=False,
                        bidirectional=True)
        
        self.mha = nn.MultiheadAttention(self.dim, num_heads = 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_outputs, _ = self.rnn(x) 
        gru_outputs = self.relu(gru_outputs)
        att_x = gru_outputs 
        attn_output, attn_output_weights = self.mha(att_x, att_x, att_x) 
        attn_output = self.relu(attn_output)
        merge_x_attn_out = torch.cat([gru_outputs, attn_output], dim=2) 
        return merge_x_attn_out

    def attention_net(self, x, query, mask=None): 
        d_k = query.size(-1)     # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n
