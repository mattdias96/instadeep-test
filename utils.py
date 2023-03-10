"""
This module contains the helper functions for
the train, predict and evaluate modules
"""
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
#from itertools import islice

from data.dataset import SequenceDataset

def loadData(num_workers:int, word2id:dict, fam2label:dict,
            seq_max_len:int, data_dir:str, batch_size:int)->dict:
    """
    Load the dataset and return the train, dev and test dataloaders

    Parameters:
        num_workers (int): Number of worker threads to use for data loading
        word2id (dict): mapping from aminoacid to id
        fam2label (dict): mapping from family to label
        seq_max_len (int): maximum sequence length
        data_dir (str): path to the data directory
        batch_size (int): the batch size

    Returns:
        dataloaders (dict): containing the train, dev and test dataloaders.
    """
    train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")
    dev_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "dev")
    test_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "test")


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloaders['dev'] = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloaders

def buildLabels(targets:pd.Series)->dict:
    """
    Creates a dictionary which maps the unique targets in the input pandas Series
    to consecutive integers starting from 1 and an additional key, specified by
    the input unknown_token, with a value of 0.Prints the number of labels present
    in the resulting dictionary.

    Parameters:
    targets (pd.Series) : A pandas Series that contains the target labels.
    Returns:
    Dict[str,int] : The created dictionary which maps the unique targets to consecutive integers.
    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    return fam2label

def buildVocab(data:list, rare_AA_count:int)->dict:
    """
    Builds a vocabulary of amino acids from a list of sequences and creates
    a mapping from AA strings to unique integers.

    Parameters:
        - data (list): a list of sequences, where each sequence is a string of
        amino acids.
        - rare_AA_count (int, optional): The number of rare AAs to remove from
        the vocabulary (defaults to 4).

    Returns:
        - word2id (dict): a dictionary that maps each unique AA in data to a unique integer.
        The integers are assigned in the order that the AAs appear in the data, starting from 2.
        Two additional special tokens, '<pad>' and '<unk>', are added to the mapping, they are
        assigned the integers 0 and 1, respectively.
    """
    # Build the vocabulary
    voc = {}
    for sequence in data:
        for aa in sequence:
            if aa not in voc:
                voc[aa] = 1
            else:
                voc[aa] += 1

    # Count the number of unique AAs and remove the rarest `rare_AA_count` AAs
    if rare_AA_count > 0:
        rare_AAs = {aa for aa, count in sorted(voc.items(), key=lambda x: x[1])[:rare_AA_count]}
        unique_AAs = sorted([aa for aa, count in voc.items() if aa not in rare_AAs])
    else:
        unique_AAs = sorted(voc.keys())

    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    return word2id

def evaluateModel(model:torch.nn.Module, test_loader:torch.utils.data.DataLoader)->float:
    """
    This function takes a PyTorch model and a test dataloader, and evaluates the model's performance
    by computing the mean accuracy over the entire test set.

    Parameters:
    - model (torch.nn.Module): A PyTorch model to be evaluated.
    - test_loader (torch.utils.data.DataLoader): A PyTorch dataloader for the test set.

    Returns:
    - float: The mean accuracy of the model over the entire test set.
    """
    accs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            acc = model.validation_step(batch, i, logging = False)
            accs.append(acc)
    return np.mean(accs)

def getPreds(model:torch.nn.Module, test_loader:torch.utils.data.DataLoader)->list:
    """
    This function takes a PyTorch model and a test dataloader, and returns the predictions of the model
    on the entire test set.

    Parameters:
    - model (torch.nn.Module): A PyTorch model to be evaluated.
    - test_loader (torch.utils.data.DataLoader): A PyTorch dataloader for the test set.

    Returns:
    - list: A list of numpy arrays containing the predictions of the model on the test set.
    """
    preds = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader)):
            x, _ = batch['sequence'], batch['target']
            y_hat = model(x)
            pred = torch.argmax(y_hat, dim=1, keepdim = True)
            preds.append(pred)
    return preds
