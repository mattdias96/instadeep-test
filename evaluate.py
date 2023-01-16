import argparse
import json

import lightning as pl
import torch

from helper import loadData, buildLabels, buildVocab, evaluateModel
from data import reader
from models.protcnn import ProtCNN

def main():

    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument("--model", type=str, default = "default_protCNN")
    parser.add_argument('--data_dir', type=str, default='', help='path to the dataset to be evaluated')
    parser.add_argument('--seq_max_len', type=int, default=120, help='maximum length of the aminoacid sequence')
    parser.add_argument('--rare_aa_count', type=int, default=5, help='number of Amino Acids to be considered rare')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')

    # Parse the command line arguments
    args = parser.parse_args()
    # Read train data files
    valid_data, valid_targets = reader("valid", args.data_dir) # change this in the reader function later
    # Define dictionary from AA strings to unique integers
    word2id = buildVocab(valid_data, args.rare_aa_count)
    # Define dictionary mapping unique targets to consecutive integers
    fam2label = buildLabels(valid_targets)
    # Define number of classes in the dataset
    num_classes = len(fam2label)
    # Retrieve pretrained model
    model = ProtCNN() # make this flexible later
    model.load_state_dict(torch.load(args.model + '.pth')) # allow user to use own model later
    # Load the data
    loader = loadData(args.num_workers, word2id, fam2label, args.seq_max_len, args.data_dir, args.batch_size)
    # Evaluate predictions
    accuracy = evaluateModel(model, loader["test"])
    # Print result
    print(accuracy)

if __name__ == "__main__":
    main()
