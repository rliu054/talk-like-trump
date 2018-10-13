import argparse
import glob

from data_processor import DataProcessor

parser = argparse.ArgumentParser(description='Generate tweets in Trump style')
parser.add_argument('--data', type=str, default='./data',
                    help='data corpus location')
args = parser.parse_args()


# -- generate dataset
file_list = glob.glob('./data/*.json')
processor = DataProcessor(file_list)
train_set, val_set, test_set = processor.generate_dataset()
