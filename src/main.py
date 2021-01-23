from typenetdataset import TypeNetDataset
from feature_extractor import FeatureExtractor

EXTRACT_FEATURES = True
SEQUENCE_SIZE = 50
DATA_ROOT_DIR = 'data'

if EXTRACT_FEATURES:
    extractor = FeatureExtractor(DATA_ROOT_DIR, SEQUENCE_SIZE)
    extractor.extract_features()

# training_data = TypeNetDataset('data', 50, True)