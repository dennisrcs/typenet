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

class TypeNetDataset(data.Dataset):
    """TAC dataset with auxiliary methods."""

    def __init__(self, root_dir, sequence_size, train=True):
        self.root_dir = root_dir
        self.sequence_size = sequence_size
        self.train = train

        # loading training data
        self.input_sequences, self.labels = self.loadSequences()

    def loadSequences(self):
        input_sequences = []
        labels = []
        
        if self.train:
            all_sequences = self.loadAllTrainingSequences()
        else:
            all_sequences = self.loadAllTestSequences()

        for i in range(0, len(all_sequences)):
            client_sequences = all_sequences[i]

            # building genuine samples + imposter triplets
            for j in range(0, len(client_sequences)):
                # genuine user sequence
                rand1 = random.randrange(0, len(client_sequences)) 
                client_sequence1 = client_sequences[j]
                client_sequence2 = client_sequences[rand1]

                # imposter sequence
                imposter_index = i
                while imposter_index == i:
                    imposter_index = random.randrange(0, len(all_sequences))
                rand2 = random.randrange(0, len(all_sequences[imposter_index]))
                imposter_sequence = all_sequences[imposter_index][rand2]

                # building triplet
                sample = [Variable(torch.Tensor(client_sequence1)), Variable(torch.Tensor(client_sequence2)), Variable(torch.Tensor(imposter_sequence))]
                
                input_sequences.append(torch.stack(sample))
                labels.append([1,0])

        return input_sequences, labels
        
    def loadAllTrainingSequences(self):
        all_sequences = []
        
        parts = np.concatenate((np.arange(2,20), [21], np.arange(23,27)))
        for p in parts:
            # train_x_relaxed_1, train_y_relaxed_1 = self.loadData(str(p), EState.EASY, '1')
            # train_x_relaxed_2, train_y_relaxed_2 = self.loadData(str(p), EState.EASY, '2')
            # train_x_relaxed_3, train_y_relaxed_3 = self.loadData(str(p), EState.EASY, '3')

            # train_x_relaxed_sequence_1, _ = self.transformIntoSequences(train_x_relaxed_1, train_y_relaxed_1)
            # train_x_relaxed_sequence_2, _ = self.transformIntoSequences(train_x_relaxed_2, train_y_relaxed_2)
            # train_x_relaxed_sequence_3, _ = self.transformIntoSequences(train_x_relaxed_3, train_y_relaxed_3)

            # client_sequences = np.concatenate((train_x_relaxed_sequence_1, train_x_relaxed_sequence_2, train_x_relaxed_sequence_3))
            client_sequences = []
            all_sequences.append(client_sequences)

        return all_sequences

    def loadAllTestSequences(self):
        all_sequences = []
        parts = np.concatenate((np.arange(2,20), [21], np.arange(23,27)))
        for p in parts:
            #train_x_relaxed_1, train_y_relaxed_1 = self.loadData(str(p), EState.EASY, '4')
            #train_x_relaxed_sequence_1, _ = self.transformIntoSequences(train_x_relaxed_1, train_y_relaxed_1)
            train_x_relaxed_sequence_1 = []
            all_sequences.append(train_x_relaxed_sequence_1)

        return all_sequences

    def loadImposterSequences(self, participant_number, state, session_num):
        imposter_sequences = []

        for p in range(2, 27):
            if p is not int(participant_number) and p is not 20 and p is not 22:
                data_x, data_y = self.loadData(participant_number, state, session_num)
                sequences_x, _ = self.transformIntoSequences(data_x, data_y)

                if len(imposter_sequences) == 0:
                    imposter_sequences = sequences_x
                else:
                    imposter_sequences = np.concatenate((imposter_sequences, sequences_x))

        return imposter_sequences

    def createInputData(self, relaxed_sequences, stressed_sequences):
        input_sequences = []
        labels = []

        for sequence1 in relaxed_sequences:
            rand1 = random.randrange(0, len(relaxed_sequences)) 
            rand2 = random.randrange(0, len(stressed_sequences))

            sample = [Variable(torch.Tensor(sequence1)), Variable(torch.Tensor(relaxed_sequences[rand1])), Variable(torch.Tensor(stressed_sequences[rand2]))]
            
            input_sequences.append(torch.stack(sample))
            labels.append([1,0])

        for sequence1 in stressed_sequences:
            rand1 = random.randrange(0, len(stressed_sequences))
            rand2 = random.randrange(0, len(relaxed_sequences)) 
            
            sample = [Variable(torch.Tensor(sequence1)), Variable(torch.Tensor(stressed_sequences[rand1])), Variable(torch.Tensor(relaxed_sequences[rand2]))]
            
            input_sequences.append(torch.stack(sample))
            labels.append([1,0])

        return input_sequences, labels

    def transformIntoSequences(self, x, y):
        sequences_x = [x[i:i + self.sequence_size] for i in range(0, len(x) - self.sequence_size)]
        sequences_y = [y[i:i + self.sequence_size] for i in range(0, len(y) - self.sequence_size)]

        return sequences_x, sequences_y

    def loadData(self, participant_num, state, session_num):
        session_type = 'E'
        # if state == EState.EASY:
        #     session_type = 'E'
        # else:
        #     session_type = 'H'

        # loading labels
        filename = self.root_dir + 'dataset_11d/P' + participant_num + '/Y_S' + session_num + session_type + '.csv'
        dataframe = pd.read_csv(filename, names=['H1','H2','DD','UD','UU','DU','DX','DY','EC','PREV_KEYCODE','CUR_KEYCODE'])

        # prev_keycode = self.parse_keycode(dataframe['PREV_KEYCODE'])
        # cur_keycode = self.parse_keycode(dataframe['CUR_KEYCODE'])

        # splitting data
        inputs = np.column_stack((dataframe['H1'],
                                  dataframe['H2'],
                                  dataframe['DD'],
                                  dataframe['UD'],
                                  dataframe['UU'],
                                  dataframe['DU'],
                                  dataframe['DX'],
                                  dataframe['DY'],
                                  dataframe['EC'],
                                  dataframe['PREV_KEYCODE'],
                                  dataframe['CUR_KEYCODE']))

        # label 0 if easy sessions, 0 for hard sessions
        if session_type == 'E':
            outputs = np.zeros(inputs.shape[0], dtype=np.int64)
        else:
            outputs = np.ones(inputs.shape[0], dtype=np.int64)

        outputs = to_categorical(outputs, 2)
        outputs = np.array(outputs, dtype=np.float64)

        data_double = inputs[:,0:8]
        data_keys = inputs[:,9:11]

        # normalizing
        # data_double = (data_double - np.mean(data_double, 0)) / np.std(data_double, 0)
        data_keys = data_keys / 76

        # inputs[:,0:8] = data_double
        inputs[:,9:11] = data_keys

        # indices = [2,3,4,5,9]
        # indices = [6,7,8]
        # return inputs[:,indices], outputs
        return inputs, outputs

    def parse_keycode(self, keycodes):
        res = np.zeros((len(keycodes), 77))
        for i, key in enumerate(keycodes):
            res[i] = self.from_int_to_1_hot(key)

        return res
    
    def from_int_to_1_hot(self, key):
        res = np.zeros((77), dtype=np.int64)
        res[key-1] = 1

        return res

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return self.input_sequences[index], self.labels[index]
