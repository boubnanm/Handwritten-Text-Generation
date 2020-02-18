from torch.utils.data import DataLoader
import os
import torch
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import svgwrite
from IPython.display import SVG, display
from tqdm import tqdm


class HandwritingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_length=300, split='train', limit=500, scale_factor=20, batch_size=50, test=False):
        super().__init__()

        self.seq_length = seq_length
        self.limit = limit
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.test = test

        self.data_dir = data_dir
        self.strokes_dir = os.path.join(data_dir, "lineStrokes")
        self.labels_dir = os.path.join(data_dir, "ascii")
        self.train_strokes_file = os.path.join(data_dir, "train_strokes.cpkl")
        self.train_labels_file = os.path.join(data_dir, "train_labels.cpkl")
        self.val_strokes_file = os.path.join(data_dir, "val_strokes.cpkl")
        self.val_labels_file = os.path.join(data_dir, "val_labels.cpkl")

        if not os.path.exists(self.train_strokes_file) or not os.path.exists(self.train_labels_file) \
                or not os.path.exists(self.val_strokes_file) or not os.path.exists(self.val_labels_file):
            self.preprocess_data()

        if split=="train":
            with open(self.train_strokes_file, "rb") as f:
                self.strokes = pickle.load(f)
            with open(self.train_labels_file, "rb") as f:
                self.labels = pickle.load(f)

        elif split=="val":
            with open(self.val_strokes_file, "rb") as f:
                self.strokes = pickle.load(f)
            with open(self.val_labels_file, "rb") as f:
                self.labels = pickle.load(f)


    def preprocess_data(self):
        if not os.path.exists(self.strokes_dir) or not os.path.exists(self.labels_dir):
            raise ValueError("Please download lineStrokes and ascii data directories from the IAM online database and put them in the data directory")
        def getSentence(txt_path, sent_id):
            sents = []
            ESCAPE_CHAR = '~!@#$%^&*()_+{}:"<>?`-=[];\',./|\n'
            for line in open(txt_path, 'r'):
                for char in ESCAPE_CHAR:
                    line = line.replace(char, '').strip()
                sents.append(line)
            try: return sents[sents.index('CSR')+2:][sent_id]
            except: return ""

        # function to read each individual xml file
        def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()
            result = []
            x_offset = 1e20
            y_offset = 1e20
            y_height = 0
            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))
                y_height = max(y_height, float(root[0][i].attrib['y']))
            y_height -= y_offset
            x_offset -= 100
            y_offset -= 100
            for stroke in root[1].findall('Stroke'):
                points = []
                for point in stroke.findall('Point'):
                    points.append([float(point.attrib['x']) - x_offset, float(point.attrib['y']) - y_offset])
                result.append(points)
            return result

        # converts a list of arrays into a 2d numpy int16 array
        def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)
            prev_x = 0
            prev_y = 0
            counter = 0
            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if (k == (len(stroke[j]) - 1)):  # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            stroke_data = np.minimum(stroke_data, self.limit)
            stroke_data = np.maximum(stroke_data, -self.limit)
            stroke_data = np.array(stroke_data, dtype=np.float32)
            stroke_data[:, 0:2] /= self.scale_factor
            return stroke_data

        strokes = []
        labels = []

        print("Preprocessing..")
        for dirName, _, fileList in tqdm(os.walk(os.path.join(self.data_dir, "lineStrokes"))):
            for filename in fileList:
                if os.path.splitext(filename)[1] == '.xml':
                    file_path = os.path.join(dirName, filename)
                    #print('processing {}'.format(file_path))

                    txt_file = "-".join(filename.split("-")[:-1]) + ".txt"
                    txt_file_path = os.path.join(dirName.replace("lineStrokes", "ascii"), txt_file)
                    sent_id = int(os.path.splitext(filename)[0][-2:]) - 1

                    sentence = getSentence(txt_file_path, sent_id)
                    stroke = convert_stroke_to_array(getStrokes(file_path))

                    if sentence and stroke.shape[0]>self.seq_length+1:
                        n_seq = stroke.shape[0]//(self.seq_length+2)
                        strokes.append(stroke)
                        labels.append(sentence)
                        while np.random.random() >= (1.0 / float(n_seq)):
                            strokes.append(stroke)
                            labels.append(sentence)

        strokes_train, strokes_val, labels_train, labels_val = train_test_split(strokes, labels, test_size=0.05, random_state=111)

        with open(self.train_strokes_file, "wb") as f:
            pickle.dump(strokes_train, f, protocol=2)

        with open(self.val_strokes_file, "wb") as f:
            pickle.dump(strokes_val, f, protocol=2)

        with open(self.train_labels_file, "wb") as f:
            pickle.dump(labels_train, f, protocol=2)

        with open(self.val_labels_file, "wb") as f:
            pickle.dump(labels_val, f, protocol=2)


    def __len__(self):
        return len(self.strokes)


    def __getitem__(self, idx):

        start = np.random.randint(0, len(self.strokes[idx]) - self.seq_length - 2 + 1)
        if self.test:
            start = 0

        stroke = torch.from_numpy(self.strokes[idx][start:start+self.seq_length]).type(torch.FloatTensor)
        target_stroke = torch.from_numpy(self.strokes[idx][start+1:start+self.seq_length + 1]).type(torch.FloatTensor)

        sentence = self.labels[idx]

        return stroke, target_stroke, sentence



def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)



def draw_strokes(data, factor=10, svg_filename='sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1

    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)

    command = "m"

    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "

    the_color = "black"
    stroke_width = 1

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

    dwg.save()
    display(SVG(dwg.tostring()))