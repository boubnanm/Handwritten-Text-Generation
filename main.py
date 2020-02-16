from utils import *
from model import Model
from torch.utils.data import DataLoader
import torch.optim as optim

epochs = 100
lr = 1e-4
batch_size = 50

train_loader = DataLoader(HandwritingDataset(data_dir="./data", split="train"), batch_size=batch_size, shuffle=True)
val_loader = HandwritingDataset(data_dir="./data", split="val")

lstm_model = Model(seq_length=300)

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

optimizer = optim.Adam(lstm_model.parameters(), lr=lr)

batch_interval = ((len(train_loader.dataset)/batch_size)+1)//5


for epoch in range(epochs):
    for batch_idx, (stroke, target_stroke, sent) in enumerate(train_loader):
        eos, pi, mus_1, mus_2, sigmas_1, sigmas_2, rhos = lstm_model(stroke)

        x1, x2, x3 = target_stroke[:,:,0].unsqueeze(2), target_stroke[:,:,1].unsqueeze(2), target_stroke[:,:,2].unsqueeze(2)

        epsilon = 1e-20

        norm_1, norm_2 = (x1 - mus_1).pow(2), (x2 - mus_2).pow(2)
        var_1, var_2 = sigmas_1.pow(2), sigmas_2.pow(2)
        co_var = 2 * rhos * (x1 - mus_1) * (x2 - mus_2) / (sigmas_1 * sigmas_2)
        Z = norm_1/var_1 + norm_2/var_2 - co_var
        normal = torch.exp(-Z/(2*(1-rhos.pow(2)))) /(2*torch.FloatTensor([np.pi])*sigmas_1*sigmas_2*torch.sqrt((1-rhos.pow(2))))
        mixture = torch.sum(normal*pi,dim=2)

        log_mixture = torch.log(torch.max(mixture, epsilon*torch.ones_like(mixture))).unsqueeze(2)
        log_bernoulli = x3*torch.log(eos.unsqueeze(2)) + (1-x3)*torch.log(1-eos.unsqueeze(2))

        #log_loss = torch.mean(torch.sum(-log_mixture - log_bernoulli, dim=(1,2)))
        log_loss = torch.mean(-log_mixture - log_bernoulli)

        optimizer.zero_grad()
        log_loss.backward()

        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 10)
        optimizer.step()

        if (batch_idx+1) % batch_interval == 0 or batch_idx==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, (batch_idx+1) * len(stroke), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), log_loss.data.item()))




