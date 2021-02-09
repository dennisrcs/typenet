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
from numpy import sqrt, floor

class TypeNetDataset(data.Dataset):
    """TAC dataset with auxiliary methods."""

    def __init__(self, root_dir, enrollment_sequences, parts):
        self.root_dir = os.path.join(root_dir, 'parsed')
        self.num_sequences = enrollment_sequences
        self.parts = parts
        self.num_participants = len(parts)
        
        # genuine numbers
        self.genuine_per_participant = int((self.num_sequences * (self.num_sequences - 1)) / 2)
        self.genuine_total = self.genuine_per_participant * self.num_participants
        
        # impostor numbers
        self.impostor_per_participant = self.num_sequences * self.num_sequences * (self.num_participants - 1)
        self.impostor_total = self.impostor_per_participant * self.num_participants

        # number of samples
        self.dataset_size = self.genuine_total + self.impostor_total

        # loading training data
        self.all_sequences = self.loadAllSequences(parts)
        
    def loadAllSequences(self, parts):
        all_sequences = []

        flist = []
        for p in pathlib.Path(self.root_dir).iterdir():
            if not p.is_file():
                flist.append(p)

        flist.sort()

        index = 0
        while len(all_sequences) < len(parts):
            if self.participantHasAllSequences(flist[index]):
                client_sequences = self.loadSequencesPerParticipant(flist[index])
                all_sequences.append(client_sequences)
            else:
                print("could not load: " + str(flist[index]))
            index = index + 1

        return all_sequences

    def participantHasAllSequences(self, part_dir):
        exists = True
        for session in np.arange(1, self.num_sequences + 1):
            filename = os.path.join(part_dir, str(session) + '.csv')
            exists &= os.path.exists(filename)

        return exists

    def loadSequencesPerParticipant(self, part_dir):
        sequences = []
        for session in np.arange(1, self.num_sequences + 1):
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

    def getGenuinePair(self, sample_index):
        part_index = sample_index // self.genuine_per_participant
        index = sample_index % self.genuine_per_participant
        
        i = self.num_sequences - 2 - floor(sqrt(-8*index + 4*self.num_sequences*(self.num_sequences-1)-7)/2.0 - 0.5)
        j = index + i + 1 - self.num_sequences*(self.num_sequences-1)/2 + (self.num_sequences-i)*((self.num_sequences-i)-1)/2

        sequence1 = self.all_sequences[part_index][int(i)]
        sequence2 = self.all_sequences[part_index][int(j)]

        sample = [Variable(torch.Tensor(sequence1)), Variable(torch.Tensor(sequence2))]
        stacked = torch.stack(sample)

        return stacked
    
    def getImposterPair(self, sample_index):
        index = sample_index - self.genuine_total

        firstPart = index // self.impostor_per_participant
        remainder = index % self.impostor_per_participant

        firstPartSequence = remainder % self.num_sequences
        remainder = remainder // self.num_sequences

        secondParticipant = remainder // self.num_sequences
        secondPartSequence = remainder % self.num_sequences

        remaining_participants_sequences = self.all_sequences[:firstPart] + self.all_sequences[firstPart+1:]

        sequence1 = self.all_sequences[firstPart][firstPartSequence]
        sequence2 = remaining_participants_sequences[secondParticipant][secondPartSequence]

        sample = [Variable(torch.Tensor(sequence1)), Variable(torch.Tensor(sequence2))]
        stacked = torch.stack(sample)

        return stacked

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        if index % 2 == 0:
            rand_index = random.randrange(self.genuine_total)
            data = self.getGenuinePair(rand_index)
            return data, 1
        else:
            rand_index = random.randrange(self.impostor_total)
            data = self.getImposterPair(index)
            return data, 0            
