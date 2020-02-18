from utils import *
from model import Model


#loader = HandwritingDataset(data_dir="./data", split="val", scale_factor=1, seq_length=300)

#lstm_model = Model(seq_length=300)
#lstm_model.load_state_dict(torch.load("./saves/model_11.pth"))
#stroke = lstm_model.sample(100)


# idx = np.random.choice(len(loader))
# stroke, sentence = loader.strokes[idx], loader.labels[idx]
#
# print(sentence)
# draw_strokes(stroke)

use_cuda = torch.cuda.is_available()
lstm_model = Model(seq_length=300, bidirectional=False)
load_epoch = max([int(os.path.splitext(fname)[0].split("_")[1]) for fname in os.listdir("./saves") if "model" in fname])

if use_cuda:
    lstm_model.cuda()
    lstm_model.load_state_dict(torch.load("./saves/model_{}.pth".format(load_epoch)))

else:
    lstm_model.load_state_dict(torch.load("./saves/model_{}.pth".format(load_epoch), map_location='cpu'))


stroke = lstm_model.sample(800)
draw_strokes(stroke, factor=5)

