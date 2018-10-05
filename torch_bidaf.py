import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

EMBED_SIZE = 3
HIDDEN_SIZE = 6
CONTEXT_SIZE = 22
QUERY_SIZE = 10

d = HIDDEN_SIZE

class BIDAF(nn.Module):
    def __init__(self):
        super().__init__()

        # layer 3: contextual embedding layer
        self.d = HIDDEN_SIZE
        self.layer3_context_lstm = nn.LSTM(EMBED_SIZE, self.d, bidirectional=True)
        self.layer3_context_hidden = self.init_hidden()
        self.layer3_query_lstm = nn.LSTM(EMBED_SIZE, self.d, bidirectional=True)
        self.layer3_query_hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2, 1, self.d), torch.zeros(2, 1, self.d))

    # input size: N * EMBED_SIZE
    def forward(self, context_word_vec, query_word_vec=None):
        context_wv = context_word_vec.view(context_word_vec.size(0), 1, -1)
        print(context_wv)

        out, self.layer3_context_hidden = self.layer3_context_lstm(context_wv, self.layer3_context_hidden)
        print('out:\n', out)
        print('hidden:\n', self.layer3_context_hidden)



### test test
model = BIDAF()
print(model)

a_context = torch.randn(5, EMBED_SIZE)
print(a_context)
print(a_context.size())
model(a_context)

