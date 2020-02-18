from utils import *
from model import Model
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=256, help='size of LSTM hidden state')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
parser.add_argument('--batch_size', type=int, default=50, help='Size of training batch')
parser.add_argument('--seq_length', type=int, default=300, help='Length of stroke sequence to train on')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--grad_clip', type=float, default=10, help='Value to clip gradients on')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')

parser.add_argument('--save_every', type=int, default=30, help='save frequency')
parser.add_argument('--model_dir', type=str, default='./saves', help='Directory path to save models in')


parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate for Adam optimizer learning rate')
parser.add_argument('--decay_every', type=int, default=5, help='Epoch frequence to decay learning rate')

parser.add_argument('--num_mixture', type=int, default=20, help='Number of gaussian mixtures')
parser.add_argument('--stroke_scale', type=float, default=20, help='Factor to scale raw strokes data down by')

args = parser.parse_args()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


use_cuda = torch.cuda.is_available()


train_loader = DataLoader(HandwritingDataset(data_dir="./data", split="train", seq_length=args.seq_length, batch_size=args.batch_size, scale_factor=atgs.stroke_scale),
                          batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(HandwritingDataset(data_dir="./data", split="val", seq_length=args.seq_length, batch_size=args.batch_size),
                        batch_size=args.batch_size, shuffle=False)

lstm_model = Model(seq_length=args.seq_length, bidirectional=False, num_mixtures=args.num_mixture, hidden_size=256)

if use_cuda:
    lstm_model.cuda()

optimizer = optim.Adam(lstm_model.parameters(), lr=args.lr)

batch_interval = ((len(train_loader.dataset) / args.batch_size) + 1) // 5

for epoch in range(args.epochs):
    for batch_idx, (stroke, target_stroke, sent) in enumerate(train_loader):

        loss = lstm_model.compute_loss(stroke, target_stroke)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 10)
        optimizer.step()

        if (batch_idx + 1) % batch_interval == 0 or batch_idx == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, (batch_idx + 1) * len(stroke), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))

    val_loss = []
    for batch_idx, (stroke, target_stroke, _) in enumerate(val_loader):
        val_loss.append(lstm_model.compute_loss(stroke, target_stroke).data.item())
    print("Validation Loss: {:.6f}".format(np.mean(val_loss)))

    if (epoch + 1) % args.save_every == 0:
        print("Saving model to ./saves/model_{}.pth".format(epoch + 1))
        torch.save(lstm_model.state_dict(), os.path.join(args.model_dir,"model_{}.pth".format(epoch + 1)))

    if (epoch + 1) % args.decay_every == 0:
        for g in optimizer.param_groups:
            lr = g['lr']
            g['lr'] = g['lr'] * args.decay_rate
        print("Learning rate decay from {:.4f} to {:.4f}".format(lr, g['lr']))

    print("\n")
