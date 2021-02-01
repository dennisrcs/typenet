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

class TypeNetDataset(data.Dataset):
    """TAC dataset with auxiliary methods."""

    def __init__(self, root_dir, enrollment_sequences, parts):
        self.root_dir = os.path.join(root_dir, 'parsed')
        self.enrollment_sequences = enrollment_sequences
        self.parts = parts

        # loading training data
        self.genuine_pairs, self.impostor_pairs = self.loadSequences()

    def loadSequences(self):
        genuine_pairs = []
        impostor_pairs = []

        all_sequences = self.loadAllSequences(self.parts)

        for p1 in range(0, len(all_sequences)):
            client_sequences = all_sequences[p1]

            # building genuine pairs
            for i in range(0, len(client_sequences)):
                for j in range(0, len(client_sequences)):
                    if i == j:
                        continue

                    client_sequence1 = client_sequences[i]
                    client_sequence2 = client_sequences[j]

                    # building pair
                    sample = [Variable(torch.Tensor(client_sequence1)), Variable(torch.Tensor(client_sequence2))]
                    genuine_pairs.append(torch.stack(sample))

            # building imposter pairs
            for p2 in range(0, len(all_sequences)):
                if p1 == p2:
                    continue

                imposter_sequences = all_sequences[p2]

                # building genuine pairs
                for i in range(0, len(client_sequences)):
                    for j in range(0, len(imposter_sequences)):
                        client_sequence = client_sequences[i]
                        imposter_sequence = imposter_sequences[j]

                        # building pair
                        sample = [Variable(torch.Tensor(client_sequence)), Variable(torch.Tensor(imposter_sequence))]
                        impostor_pairs.append(torch.stack(sample))

        return genuine_pairs, impostor_pairs
        
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
        for session in np.arange(1, self.enrollment_sequences + 1):
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
        return len(self.genuine_pairs) + len(self.impostor_pairs)

    def __getitem__(self, index):
        if index < len(self.genuine_pairs):
            return self.genuine_pairs[index], 1
        else:
            return self.impostor_pairs[index - len(self.genuine_pairs)], 0

        return sample
