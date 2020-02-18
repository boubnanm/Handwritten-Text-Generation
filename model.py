import torch.nn as nn
import torch
import numpy as np

class Model(nn.Module):
    def __init__(self, seq_length=300, num_mixtures=20, bidirectional=False, hidden_size=256, num_layers=2, stroke_factor=20):
        super(Model, self).__init__()

        self.seq_length = seq_length
        self.num_mixtures = num_mixtures
        self.hidden_size = hidden_size
        self.stroke_factor = stroke_factor

        self.blstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        self.linear = nn.Linear(hidden_size*(1+bidirectional), num_mixtures*6+1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out, _ = self.blstm(x)

        out = self.linear(out.contiguous())

        eos = self.sigmoid(out[:, :, 0])
        pi = self.softmax(out[:, :, 1:1+self.num_mixtures])
        mus_1 = out[:, :, 1 + self.num_mixtures:1 + 2 * self.num_mixtures]
        mus_2 = out[:, :, 1 + 2 * self.num_mixtures:1 + 3 * self.num_mixtures]
        sigmas_1 = torch.exp(out[:, :, 1 + 3 * self.num_mixtures:1 + 4 * self.num_mixtures])
        sigmas_2 = torch.exp(out[:, :, 1 + 4 * self.num_mixtures:1 + 5 * self.num_mixtures])
        rhos = self.tanh(out[:, :, 1 + 5 * self.num_mixtures:])

        return eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos

    def step_forward(self, x, h_n, c_n):

        out, (h_n, c_n) = self.blstm(x, (h_n, c_n))

        out = self.linear(out.contiguous())

        eos = self.sigmoid(out[:, :, 0])
        pi = self.softmax(out[:, :, 1:1+self.num_mixtures])
        mus_1 = out[:, :, 1 + self.num_mixtures:1 + 2 * self.num_mixtures]
        mus_2 = out[:, :, 1 + 2 * self.num_mixtures:1 + 3 * self.num_mixtures]
        sigmas_1 = torch.exp(out[:, :, 1 + 3 * self.num_mixtures:1 + 4 * self.num_mixtures])
        sigmas_2 = torch.exp(out[:, :, 1 + 4 * self.num_mixtures:1 + 5 * self.num_mixtures])
        rhos = self.tanh(out[:, :, 1 + 5 * self.num_mixtures:])

        return eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos, h_n, c_n

    def compute_loss(self, stroke, target_stroke):
        if torch.cuda.is_available():
            stroke, target_stroke = stroke.cuda(), target_stroke.cuda()

        eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos = self.forward(stroke)

        x1, x2, x3 = target_stroke[:, :, 0].unsqueeze(2), target_stroke[:, :, 1].unsqueeze(2), target_stroke[:, :,
                                                                                               2].unsqueeze(2)

        epsilon = 1e-20

        if torch.cuda.is_available():
            cte_pi = torch.FloatTensor([np.pi]).cuda()
        else:
            cte_pi = torch.FloatTensor([np.pi])

        norm_1, norm_2 = (x1 - mus_1).pow(2), (x2 - mus_2).pow(2)
        var_1, var_2 = sigmas_1.pow(2), sigmas_2.pow(2)
        co_var = 2 * rhos * (x1 - mus_1) * (x2 - mus_2) / (sigmas_1 * sigmas_2)
        Z = norm_1 / var_1 + norm_2 / var_2 - co_var
        normal = torch.exp(-Z / (2 * (1 - rhos.pow(2)))) / (
                    2 * cte_pi * sigmas_1 * sigmas_2 * torch.sqrt((1 - rhos.pow(2))))
        mixture = torch.sum(normal * pi, dim=2)

        log_mixture = torch.log(torch.max(mixture, epsilon * torch.ones_like(mixture))).unsqueeze(2)
        log_bernoulli = x3 * torch.log(eos.unsqueeze(2)) + (1 - x3) * torch.log(1 - eos.unsqueeze(2))

        log_loss = torch.mean(-log_mixture - log_bernoulli)

        return log_loss


    def sample(self, length=100):

        prev_x = torch.zeros(1, 1, 3).type(torch.FloatTensor)
        prev_x[0,0,2] = 1
        output_strokes = np.zeros((length, 3))
        h_n = torch.zeros(2, 1, self.hidden_size).type(torch.FloatTensor)
        c_n = torch.zeros(2, 1, self.hidden_size).type(torch.FloatTensor)

        if torch.cuda.is_available():
            h_n, c_n = h_n.cuda(), c_n.cuda()

        for i in range(length):
            with torch.no_grad():
                if torch.cuda.is_available():
                    prev_x = prev_x.cuda()

                eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos, h_n, c_n = self.step_forward(prev_x, h_n, c_n)

                if torch.cuda.is_available():
                    eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos = eos.cpu(), pi.cpu(), mus_1.cpu(), mus_2.cpu(), sigmas_1.cpu(), sigmas_2.cpu(), rhos.cpu()

                idx = np.argmax(np.random.multinomial(1, np.array(pi).flatten(), 1))

                mu_1, mu_2 = np.array(mus_1).flatten()[idx], np.array(mus_2).flatten()[idx]
                s1, s2 = np.array(sigmas_1).flatten()[idx], np.array(sigmas_2).flatten()[idx]
                rho = np.array(rhos).flatten()[idx]

                mean = [mu_1, mu_2]
                cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]

                x1, x2 = np.random.multivariate_normal(mean, cov, 1).flatten()
                eos_ = np.random.binomial(1, float(eos))

                prev_x = torch.FloatTensor([x1, x2, eos_]).view(1,1,3)

                output_strokes[i] = np.array([x1,x2,eos_])

        output_strokes[:,0:2] *= self.stroke_factor

        return output_strokes




