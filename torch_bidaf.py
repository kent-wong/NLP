import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import webqa

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

EMBED_SIZE = 256
HIDDEN_SIZE = 256

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

        # layer 5: modeling layer
        self.modeling_layer_lstm = nn.LSTM(self.d*8, self.d, num_layers=2, bidirectional=True)
        
        # layer 6: output layer
        self.output_layer_w_p1 = nn.Linear(self.d*10, 1)
        self.output_layer_w_p2 = nn.Linear(self.d*10, 1)
        self.output_layer_lstm = nn.LSTM(self.d*2, self.d, bidirectional=True)

    def init_hidden(self, dim0=1):
        return (torch.zeros(dim0, 1, self.d, device=device), torch.zeros(dim0, 1, self.d, device=device))

    # h size: 2d
    # U size: 2d * J
    # return size: 1 * J
    def alpha(self, h, U):
        J = U.size(1)
        h = h.unsqueeze(0)
        H_matrix = [h] * J
        H_matrix = torch.cat(H_matrix, dim=0)
        #print(H_matrix)
        #print(U.t())
        #print(h*U.t())

        # size: J * 6d
        concated_matrix = torch.cat([H_matrix, U.t(), h*U.t()], dim=1)
        #print(concated_matrix)

        # out size: J * 1
        out = self.layer4_w_s(concated_matrix)
        #print(out.t())

        return out.t()

    # input size: S * EMBED_SIZE
    def forward(self, context_word_vec, query_word_vec):
        T = context_word_vec.size(0)
        context_wv = context_word_vec.view(T, 1, -1)
        #print(context_wv)

        out, hidden = self.layer3_context_lstm(context_wv, self.init_hidden(2))
        #print('out:\n', out)
        #print('hidden:\n', self.layer3_context_hidden)

        # H size: 2d * T
        H = out.view(T, -1).t()
        #print('H:\n', H)

        J = query_word_vec.size(0)
        context_wv = query_word_vec.view(J, 1, -1)
        #print(context_wv)

        out, hidden = self.layer3_query_lstm(context_wv, self.init_hidden(2))
        #print('out:\n', out)

        # U size: 2d * J
        U = out.view(J, -1).t()
        #print('U:\n', U)


        # calculating similarity matrix 'S'
        #for t in len(T):
        #self.alpha(H[:, 0], U)
        S_rows = [self.alpha(H[:, t], U) for t in range(T)]
        S = torch.cat(S_rows, dim=0)
        #print('S:\n', S)

        # 计算Context_to_query Attention, size: 2d * T 
        S_softmaxed = F.softmax(S, dim=1)
        UU = torch.mm(S_softmaxed, U.t()).t()
        #print('UU:\n', UU)

        # 计算Query_to_context Attention
        b, _ = torch.max(S, dim=1, keepdim=True)
        b = F.softmax(b, dim=0)
        hh = torch.mm(H, b)
        HH = torch.cat([hh]*T, dim=1)

        G = torch.cat([H, UU, H*UU, H*HH], dim=0)
        #print(G)
        #print(G.size())


        ## 计算Modeling Layer
        M, hidden = self.modeling_layer_lstm(G.t().view(T, 1, -1), self.init_hidden(4))


        ## 计算Output Layer
        # 计算答案的起始位置概率分布
        out_matrix = torch.cat([G, M.squeeze().t()], dim=0)
        #print(out_matrix)
        p1 = self.output_layer_w_p1(out_matrix.t())
        p1 = p1.t()
        #print(p1)
        p1_proba = F.log_softmax(p1, dim=1)
        #print(p1_proba)

        # 计算答案的结束位置概率分布
        M2, hidden = self.output_layer_lstm(M, self.init_hidden(2))
        out_matrix = torch.cat([G, M2.squeeze().t()], dim=0)
        p2 = self.output_layer_w_p2(out_matrix.t())
        p2 = p2.t()
        #print(p2)
        p2_proba = F.log_softmax(p2, dim=1)
        #print(p2_proba)

        out = torch.cat([p1_proba, p2_proba], dim=0)
        #print(out)
        return out


## train
model = BIDAF()
print(model)
if torch.cuda.is_available():
    model.cuda()
print(model)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

EPOCHS = 1
for epoch in range(EPOCHS):
    for query, context, target in webqa.load_qa():
        query = query.view(query.size(0), 1, -1).to(device)
        context = context.view(context.size(0), 1, -1).to(device)
        target = target.to(device)

        optimizer.zero_grad()
        out = model(context, query)
        #print('model output:\n', out)

        loss = loss_function(out, target)
        print('loss:\n', loss)

        loss.backward()
        optimizer.step()
