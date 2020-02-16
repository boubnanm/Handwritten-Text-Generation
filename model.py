import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class Model(nn.Module):
    def __init__(self, seq_length=100, num_mixtures=20, bidirectional=True, hidden_size=256):
        super(Model, self).__init__()

        self.seq_length = seq_length
        self.num_mixtures = num_mixtures

        self.blstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=2, bidirectional=bidirectional, dropout=0.2, batch_first=True)

        self.linear = nn.Linear(hidden_size*(1+bidirectional), num_mixtures*6+1)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        out, _ = self.blstm(x)

        #batch, seq_len, num_directions*hidden_size)

        out = self.linear(out.contiguous())

        eos = torch.sigmoid(out[:, :, 0])
        pi = F.softmax(out[:, :, 1:1+self.num_mixtures], dim=2)
        mus_1 = out[:, :, 1 + self.num_mixtures:1 + 2 * self.num_mixtures]
        mus_2 = out[:, :, 1 + 2 * self.num_mixtures:1 + 3 * self.num_mixtures]
        sigmas_1 = torch.exp(out[:, :, 1 + 3 * self.num_mixtures:1 + 4 * self.num_mixtures])
        sigmas_2 = torch.exp(out[:, :, 1 + 4 * self.num_mixtures:1 + 5 * self.num_mixtures])
        rhos = torch.tanh(out[:, :, 1 + 5 * self.num_mixtures:])

        # print("bernoulli", eos.shape)
        # print("pi", pi.shape)
        # print("mu1", mus_1.shape)
        # print("mu2", mus_2.shape)
        # print("var1", vars_1.shape)
        # print("var2", vars_2.shape)
        # print("corr", corrs.shape)

        return eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos


    def sample(self, length=100):

        prev_x = torch.zeros(1, 1, 3).type(torch.FloatTensor)
        prev_x[0,0,2] = 1
        output_strokes = np.zeros((length, 3))

        for i in range(length):
            with torch.no_grad():
                eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos = self.forward(prev_x)

                idx = np.argmax(np.random.multinomial(1, np.array(pi).flatten(), 1))

                mu_1, mu_2 = np.array(mus_1).flatten()[idx], np.array(mus_2).flatten()[idx]
                s1, s2 = np.array(sigmas_1).flatten()[idx], np.array(sigmas_2).flatten()[idx]
                rho = np.array(rhos).flatten()[idx]

                mean = [mu_1, mu_2]
                cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]

                x1, x2 = np.random.multivariate_normal(mean, cov, 1).flatten()
                eos = np.random.binomial(1, float(eos))

                prev_x = torch.FloatTensor((x1, x2, eos)).view(1,1,3)

                output_strokes[i] = np.array([x1,x2,eos])

        return output_strokes





