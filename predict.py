import argparse
import json
import datetime
import csv

import lightning as pl
import torch

from helper import loadData, buildLabels, buildVocab, getPreds
from data import reader
from models.protcnn import ProtCNN

def main():

    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument("--model", type=str, default = "default_protCNN") # make this flexible later
    parser.add_argument("--model_weights_file_path", type=str, required = True)
    parser.add_argument('--train_dir', type=str, default='', help='path to the dataset to be trained')
    parser.add_argument('--test_dir', type=str, default='', help='path to the dataset to be evaluated')
    parser.add_argument('--seq_max_len', type=int, default=120, help='maximum length of the aminoacid sequence')
    parser.add_argument('--rare_aa_count', type=int, default=5, help='number of Amino Acids to be considered rare')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay of the optimizer')
    parser.add_argument('--milestones', type=list, default=[5, 8, 10, 12, 14, 16, 18, 20], help='milestones of the lr scheduler')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma parameter of the lr scheduler')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker threads to use for loading the data')

    # Parse the command line arguments
    args = parser.parse_args()
    # Read train data files
    valid_data, valid_targets = reader("train", args.train_dir) # change this in the reader function later
    # Define dictionary from AA strings to unique integers
    word2id = buildVocab(valid_data, args.rare_aa_count)
    # Define dictionary mapping unique targets to consecutive integers
    fam2label = buildLabels(valid_targets)
    # Define number of classes in the dataset
    num_classes = len(fam2label)
    # Retrieve pretrained model
    model = ProtCNN(num_classes, args.lr, args.momentum, args.weight_decay, args.milestones, args.gamma) # make this flexible later
    model.load_state_dict(torch.load(args.model_weights_file_path)) # allow user to use own model later
    # Load the data
    loader = loadData(args.num_workers, word2id, fam2label, args.seq_max_len, args.train_dir, args.batch_size)
    # Create predictions
    preds = getPreds(model, loader["test"])
    preds = preds.numpy()
    # Create a unique file name with the timestamp
    now = datetime.datetime.now()
    file_name = f"data_{now.strftime('%Y-%m-%d %H-%M-%S')}.csv"
    # Save prediction in file
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(preds)

if __name__ == "__main__":
    main()