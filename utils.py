from torch.utils.data import DataLoader
import os
import torch
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import numpy as np
import pickle


class HandwritingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_length=100, split='train'):
        super().__init__()

        self.seq_length = seq_length

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
            return stroke_data.astype(np.float)

        strokes = []
        labels = []

        for dirName, _, fileList in os.walk(os.path.join(self.data_dir, "lineStrokes")):
            for filename in fileList:
                if os.path.splitext(filename)[1] == '.xml':
                    file_path = os.path.join(dirName, filename)
                    print('processing {}'.format(file_path))

                    txt_file = "-".join(filename.split("-")[:-1]) + ".txt"
                    txt_file_path = os.path.join(dirName.replace("lineStrokes", "ascii"), txt_file)
                    sent_id = int(os.path.splitext(filename)[0][-2:]) - 1

                    sentence = getSentence(txt_file_path, sent_id)

                    if sentence :
                        strokes.append(convert_stroke_to_array(getStrokes(file_path)))
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

        stroke = self.strokes[idx][:self.seq_length]
        target_stroke = self.strokes[idx][1:self.seq_length + 1]

        if stroke.shape[0]<self.seq_length:
            remain = self.seq_length - stroke.shape[0]
            stroke = torch.from_numpy(np.concatenate((stroke, np.array([0,0,1]*remain).reshape(-1,3)))).type(torch.FloatTensor)
        else:
            stroke = torch.from_numpy(stroke).type(torch.FloatTensor)

        if target_stroke.shape[0]<self.seq_length:
            remain_target = self.seq_length - target_stroke.shape[0]
            target_stroke = torch.from_numpy(np.concatenate((target_stroke, np.array([0,0,1]*remain_target).reshape(-1,3)))).type(torch.FloatTensor)
        else:
            target_stroke = torch.from_numpy(target_stroke).type(torch.FloatTensor)

        sentence = self.labels[idx]

        return stroke, target_stroke, sentence



