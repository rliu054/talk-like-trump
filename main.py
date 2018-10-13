import argparse

parser = argparse.ArgumentParser(description='Generate tweets in Trump style')
parser.add_argument('--data', type=str, default='./data',
                    help='data corpus location')
args = parser.parse_args()


if __name__ == '__main__':
    pass
