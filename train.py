"""
This module contains the main function for training a model
end-to-end and saving the trained model at the location decide
by the user
"""

import argparse

import pytorch_lightning as pl
import torch

from utils import loadData, buildLabels, buildVocab

from data.helper import reader
from models.protcnn import ProtCNN


def main():
    """
    Function processes arguments inputed by the user, preprocess data,
    train the model and saved a trained model
    """
    parser = argparse.ArgumentParser()
    # Add command line arguments
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float, default=0.9,
                        help='momentum for the optimizer')
    parser.add_argument('--weight-decay',
                        type=float, default=1e-2,
                        help='weight decay of the optimizer')
    parser.add_argument('--milestones',
                        type=list,
                        default=[5, 8, 10, 12, 14, 16, 18, 20],
                        help='milestones of the lr scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.9,
                        help='gamma parameter of the lr scheduler')
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='number of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size')
    parser.add_argument('--model-path',
                        type=str,
                        default='',
                        help='path to save the trained model')
    parser.add_argument('--train_dir',
                        type=str,
                        default='',
                        help='path to the training dataset')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='1 if GPU is used, 0 if CPU is used')
    parser.add_argument('--seq_max_len',
                        type=int,
                        default=120,
                        help='maximum length of the aminoacid sequence')
    parser.add_argument('--random_seed',
                        type=int,
                        default=0,
                        help='random seed for reproducibility')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='number of worker threads to use for loading the data')
    parser.add_argument('--rare_aa_count',
                        type=int,
                        default=5,
                        help='number of Amino Acids to be considered rare')
    parser.add_argument('--save_model_file',
                        type=str,
                        default="default",
                        help='File name to save trained model')

    # Parse the command line arguments
    args = parser.parse_args()
    # Read train data files
    train_data, train_targets = reader("train", args.train_dir)
    # Define dictionary from AA strings to unique integers
    word2id = buildVocab(train_data, args.rare_aa_count)
    # Define dictionary mapping unique targets to consecutive integers
    fam2label = buildLabels(train_targets)
    # Define number of classes in the dataset
    num_classes = len(fam2label)
    # Initialize the model
    # Create a class for this and make it flexible later
    model = ProtCNN(num_classes, args.lr, args.momentum,
                    args.weight_decay, args.milestones, args.gamma, len(word2id))
    # Set random seed
    pl.seed_everything(args.random_seed)
    # Initialize trainer module
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs)
    # Load the data
    loader = loadData(args.num_workers, word2id, fam2label,
                      args.seq_max_len, args.train_dir, args.batch_size)
    # Fit model
    trainer.fit(model, loader['train'], loader['dev'])
    # Save model
    if args.save_model_file != "default":
        torch.save(model.state_dict(), args.save_model_file)

if __name__ == "__main__":
    main()
