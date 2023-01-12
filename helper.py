import torch

from . import SequenceDataset

def loadData(num_workers, word2id, fam2label, seq_max_len, data_dir, batch_size):
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
        tuple: containing the train and dev dataloaders.
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

    return dataloaders['train'], dataloaders['dev'], dataloaders['test']
    