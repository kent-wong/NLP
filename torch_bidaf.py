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

        # layer 4: attention flow layer
        self.layer4_w_s = nn.Linear(self.d*6, 1)

    def init_hidden(self):
        return (torch.zeros(2, 1, self.d), torch.zeros(2, 1, self.d))

    # h size: 2d
    # U size: 2d * J
    # return: J * 6d
    def alpha(self, h, U):
        J = U.size(1)
        h = h.unsqueeze(0)
        H_matrix = [h] * J
        H_matrix = torch.cat(H_matrix, dim=0)
        print(H_matrix)
        print(U.t())
        print(h*U.t())

        # size: J * 6d
        concated_matrix = torch.cat([H_matrix, U.t(), h*U.t()], dim=1)
        print(concated_matrix)

        out = self.layer4_w_s(concated_matrix)
        print(out.squeeze())

    # input size: S * EMBED_SIZE
    def forward(self, context_word_vec, query_word_vec):
        T = context_word_vec.size(0)
        context_wv = context_word_vec.view(T, 1, -1)
        #print(context_wv)

        out, hidden = self.layer3_context_lstm(context_wv, self.init_hidden())
        #print('out:\n', out)
        #print('hidden:\n', self.layer3_context_hidden)

        # H size: 2d * T
        H = out.view(T, -1).t()
        print('H:\n', H)

        J = query_word_vec.size(0)
        context_wv = query_word_vec.view(J, 1, -1)
        #print(context_wv)

        out, hidden = self.layer3_query_lstm(context_wv, self.init_hidden())
        #print('out:\n', out)

        # U size: 2d * J
        U = out.view(J, -1).t()
        print('U:\n', U)


        # calculating similarity matrix 'S'
        self.alpha(H[:, 0], U)



### test test
model = BIDAF()
print(model)

a_context = torch.randn(5, EMBED_SIZE)
a_query = torch.randn(3, EMBED_SIZE)
print(a_context)
print(a_context.size())
model(a_context, a_query)

