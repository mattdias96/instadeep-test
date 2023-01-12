import argparse
import json

def main():
    
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay of the optimizer')
    parser.add_argument('--milestones', type=list, default=[5, 8, 10, 12, 14, 16, 18, 20], help='milestones of the lr scheduler')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma parameter of the lr scheduler')
    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--model-path', type=str, default='', help='path to save the trained model')
    parser.add_argument('--data_dir', type=str, default='', help='path to the training dataset')
    parser.add_argument('--gpus', type=int, default=1, help='1 if GPU is used, 0 if CPU is used')
    parser.add_argument('--seq_max_len', type=int, default=120, help='maximum length of the aminoacid sequence')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker threads to use for loading the data')

    # Parse the command line arguments
    args = parser.parse_args()

    
    # Read train data files
    train_data, train_targets = reader("train", args.dataset_path)
    # Define dictionary mapping unique targets to consecutive integers
    fam2label = buildLabels(train_targets)
    # Define number of classes in the dataset
    num_classes = len(fam2label)
    # Initialize the model
    prot_cnn = ProtCNN(num_classes)
    # Set random seed
    pl.seed_everything(args.random_seed)
    # Initialize trainer module
    trainer = pl.Trainer(gpus=parser.gpus, max_epochs=parser.epochs)
    # Load the data
    train_loader, dev_loader, test_loader = loadData(args.num_workers, word2id, fam2label, args.seq_max_len, args.data_dir, args.batch_size)
    # Fit model
    trainer.fit(prot_cnn, train_loader, dev_loader)
