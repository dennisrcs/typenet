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
import cProfile
import pstats

class TypeNetDataset(data.Dataset):
    """TAC dataset with auxiliary methods."""

    def __init__(self, root_dir, enrollment_sequences, parts):
        self.root_dir = os.path.join(root_dir, 'parsed')
        self.enrollment_sequences = enrollment_sequences
        self.parts = parts

        # loading training data
        self.genuine_pairs_path, self.impostor_pairs_path = self.loadSequences()

    def loadSequences(self):
        genuine_pairs = []
        impostor_pairs = []

        all_paths_per_participant = self.loadAllSequencesPath(self.parts)

        for p1 in range(0, len(all_paths_per_participant)):
            client_paths = all_paths_per_participant[p1]

            # building genuine pairs
            for i in range(0, len(client_paths)):
                for j in range(i+1, len(client_paths)):
                    client_path1 = client_paths[i]
                    client_path2 = client_paths[j]

                    # building pair
                    client_path_pair = [client_path1, client_path2]
                    genuine_pairs.append(client_path_pair)

            # building imposter pairs
            for p2 in range(0, len(all_paths_per_participant)):
                if p1 == p2:
                    continue

                imposter_paths = all_paths_per_participant[p2]

                # building genuine pairs
                for i in range(0, len(client_paths)):
                    for j in range(0, len(imposter_paths)):
                        client_path = client_paths[i]
                        imposter_path = imposter_paths[j]

                        # building pair
                        imposter_path_pair = [client_path, imposter_path]
                        impostor_pairs.append(imposter_path_pair)

        return genuine_pairs, impostor_pairs
        
    def loadAllSequencesPath(self, parts):
        all_sequences = []
        
        flist = []
        for p in pathlib.Path(self.root_dir).iterdir():
            if not p.is_file():
                flist.append(p)

        flist.sort()

        for p in parts:
            part_dir = flist[p]
            client_sequences = []

            for session in np.arange(1, self.enrollment_sequences + 1):
                filename = os.path.join(part_dir, str(session) + '.csv')
                client_sequences.append(filename)
            
            all_sequences.append(client_sequences)

        return all_sequences

    def loadSequencesPerParticipantPerSession(self, filenames):
        sequence1 = self.loadData(filenames[0])
        sequence2 = self.loadData(filenames[1])

        sample = [Variable(torch.Tensor(sequence1)), Variable(torch.Tensor(sequence2))]
        stacked = torch.stack(sample)

        return stacked

    def loadData(self, filename):
        dataframe = pd.read_csv(filename, header=None)

        return dataframe.to_numpy()

    def __len__(self):
        return len(self.genuine_pairs_path) + len(self.impostor_pairs_path)

    def __getitem__(self, index):
        if index < len(self.genuine_pairs_path):
            paths = self.genuine_pairs_path[index]
            data = self.loadSequencesPerParticipantPerSession(paths)
            return data, 1
        else:
            paths = self.impostor_pairs_path[index - len(self.genuine_pairs_path)]
            data = self.loadSequencesPerParticipantPerSession(paths)
            return data, 0
