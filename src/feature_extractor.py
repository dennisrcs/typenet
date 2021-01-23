import pathlib
import os
import pandas as pd
import numpy as np
import csv

class FeatureExtractor():
    def __init__(self, root_dir, sequence_size, train=True):
        self.root_dir = root_dir
        self.sequence_size = sequence_size
        
    def extract_features(self):
        raw_dir = os.path.join(self.root_dir, 'raw')
        
        flist = []
        for p in pathlib.Path(raw_dir).iterdir():
            if p.is_file():
                flist.append(p)

        for f in flist:
            try:
                self.extract_features_from_participant(f)
            except:
                print('Error extracting features from file: ' + str(f))

    def extract_features_from_participant(self, file):
        df = pd.read_csv(file, sep='\t', header=1, names=['PARTICIPANT_ID','TEST_SECTION_ID','SENTENCE','USER_INPUT','KEYSTROKE_ID','PRESS_TIME','RELEASE_TIME','LETTER','KEYCODE'])

        unique_section = set(df.TEST_SECTION_ID)
        p_id = df.PARTICIPANT_ID[df.index[0]]

        counter = 1
        for s in unique_section:
            filtered_df = df.loc[df['TEST_SECTION_ID'] == s]
            self.extract_features_from_section(filtered_df, p_id, counter)
            counter += 1


    def extract_features_from_section(self, df, participant, section):
        output = []
        for index, row in df.iterrows():
            if index == df.index[-1] or len(output) == self.sequence_size:
                break

            HL = ( df.RELEASE_TIME[index] - df.PRESS_TIME[index] ) / 1000.0
            PL = ( df.PRESS_TIME[index + 1] - df.PRESS_TIME[index] ) / 1000.0
            IL = ( df.PRESS_TIME[index + 1] - df.RELEASE_TIME[index] ) / 1000.0
            RL = ( df.RELEASE_TIME[index + 1] - df.RELEASE_TIME[index] ) / 1000.0
            KC = df.KEYCODE[index] / 255.0

            feature = [HL, PL, IL, RL, KC]

            output.append(feature)

        while len(output) < self.sequence_size:
            output.append([0,0,0,0,0])
    
        parsed_output = self.formatdata(output)

        filename = os.path.join(self.root_dir, 'parsed', str(participant), str(section) + ".csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(parsed_output)
        
    def formatdata(self, data):
        for row in data:
            yield ["%0.3f" % v for v in row]
