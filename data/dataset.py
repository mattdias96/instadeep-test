import numpy as np
import torch

from data import *


class SequenceDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset that represents a collection of sequences and their associated labels.
    sequences are one-hot encoded and padded or truncated to the `max_len`.
    
    Parameters:
        - word2id (dict): a dictionary that maps aminoacid strings to unique integers.
        - fam2label (dict): a dictionary that maps family labels to unique integers.
        - max_len (int): the maximum length of the sequences in the dataset.
        - data_path (str): the path to a file containing the data to be read and processed.
        - split (str): dataset split i.e. 'train', 'dev' or 'test'
        
    Attributes:
        - data (pd.DataFrame) : Dataframe containing the data after reading and processing
        - label (pd.DataFrame) : Dataframe containing the labels of the data
        
    """

    def __init__(self, word2id:dict, fam2label:dict, max_len:int, data_path:str, split:str):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        
        self.data, self.label = reader(split, data_path)
        
    def __len__(self):
        """
        Returns the length of the dataset, which is the number of sequences.
        """
        return len(self.data)

    def __getitem__(self, index:int):
        """
        Returns a dictionary containing the processed sequence and the target label.
        
        Parameters:
            - index (int): Index of the sequence in the dataset
            
        Returns:
            - dict: A dictionary containing the following keys,
                - 'sequence' : preprocessed, padded, one-hot encoded sequence
                - 'target' : label of the sequence
        """
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])
       
        return {'sequence': seq, 'target' : label}
    
    def preprocess(self, text:str):
        """
        Preprocess the sequence, by taking the slice of the text (sequence) of `max_len` length,
        encodes it into IDs and pads it, and one-hot encodes the sequence and returns it in the permuted form.
        
        Parameters:
            - text (str) : sequence as text
            
        Returns:
            - torch.Tensor : preprocessed and one-hot encoded sequence
        """
        seq = []
        
        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))
                
        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]
                
        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))
            
        torch.device("cpu")
        # One-hot encode    
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id), ) 

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1,0)

        return one_hot_seq