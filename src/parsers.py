import argparse


train_parser = argparse.ArgumentParser()
train_parser.add_argument(
    '-i', '--input',
    type=str,
    action='store', default="../data/input_sequences/data_for_dev",
    help="Input sequence."
)

train_parser.add_argument(
    '-b', '--batch-size',
    type=int,
    action='store', default=512,
    help="Maximum size of a batch."
)

train_parser.add_argument(
    '-o', '--output',
    type=str, action='store', default="../data/networks/",
    help="Output directory. Must not exist."
)

train_parser.add_argument(
    '-n', '--n-batches',
    type=int,
    action='store', default=5000,
    help="number of batches to train."
)


test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    '-i', '--input',
    type=str,
    action='store', default="../data/input_sequences/data_for_dev",
    help="Input sequence."
)

test_parser.add_argument(
    '-n', '--network-path',
    type=str,
    action='store', required=True,
    help="Path to network."
)

test_parser.add_argument(
    '-b', '--batch-size',
    type=int,
    action='store', default=512,
    help="Maximum size of a batch."
)

test_parser.add_argument(
    '-g', '--gamma',
    type=float,
    action='store', default=0.95,
    help="gamma."
)
