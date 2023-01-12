from data import SequenceDataset
import argparse
import json

def train_model(args):
    pass

def getData():

    # Get the data as a PyTorch dataset
    train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")
    dev_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "dev")
    test_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "test")
    pass



if __name__ == "__main__":
    