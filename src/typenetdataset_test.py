"""TAC dataset utilities."""
import torch
from torch.autograd import Variable
from torch.utils import data
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing
import numpy.testing as nptest
import random
import os
import pathlib

class TypeNetDatasetTest(data.Dataset):
    """TAC dataset with auxiliary methods."""

    def __init__(self, root_dir, parts):
        self.root_dir = os.path.join(root_dir, 'parsed')
        self.parts = parts
        self.num_test_sequences = 15

        # loading training data
        self.data = self.loadSequences()

    def loadSequences(self):
        data = []

        all_sequences = self.loadAllSequences(self.parts)

        for p1 in range(0, len(all_sequences)):
            sample = []

            client_sequences = all_sequences[p1]
            for i in range(0, len(client_sequences)):
                client_sequence = client_sequences[i]
                sample.append(Variable(torch.Tensor(client_sequence)))

            for p2 in range(0, len(all_sequences)):
                if p1 is not p2:
                    impostor_sequences = all_sequences[p2]

                    rand_index = random.randrange(0, self.num_test_sequences)
                    impostor_sequence = impostor_sequences[rand_index]
                    sample.append(Variable(torch.Tensor(impostor_sequence)))

            data.append(sample)

        return data
        
    def loadAllSequences(self, parts):
        all_sequences = []
        
        flist = []
        for p in pathlib.Path(self.root_dir).iterdir():
            if not p.is_file():
                flist.append(p)

        flist.sort()

        for p in parts:
            client_sequences = self.loadSequencesPerParticipant(flist[p])
            all_sequences.append(client_sequences)

        return all_sequences

    def loadSequencesPerParticipant(self, part_dir):
        sequences = []
        for session in np.arange(1, self.num_test_sequences + 1):
            filename = os.path.join(part_dir, str(session) + '.csv')
            dataframe = pd.read_csv(filename, names=['HL', 'PL', 'IL', 'RL', 'KC'])

            # splitting data
            inputs = np.column_stack((dataframe['HL'],
                                    dataframe['PL'],
                                    dataframe['IL'],
                                    dataframe['RL'],
                                    dataframe['KC']))

            sequences.append(inputs)

        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
