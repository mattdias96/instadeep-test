import os

import pandas as pd
from collections import Counter

data_dir = '/content/drive/MyDrive/InstaDeep/random_split' # this will be chosen by the user

def reader(partition:str, data_path:str) -> Tuple[pd.Series, pd.Series]:
    """
    Reads files from a specified partition within a given data path, loads the contents of those files into a Pandas DataFrame,
    concatenates all the dataframe into single one, and finally returns 2 specific columns of
    the dataframe : 'sequence' (regressors) and 'family_accession' (target).
    
    Parameters:
    partition (str) : The name of the partition to read files from
    data_path (str) : The path to the data
    
    Returns:
    Tuple : (pandas.Series, pandas.Series) - containing the 'sequence' and 'family_accession' columns of the concatenated DataFrame
    """
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)        

    return all_data["sequence"], all_data["family_accession"]

def build_labels(targets:pd.Series)->Dict[str,int]:
    """
    Creates a dictionary which maps the unique targets in the input pandas Series to consecutive integers starting from 1 and an additional key, specified by the input unknown_token, with a value of 0.
    Prints the number of labels present in the resulting dictionary.

    Parameters:
    targets (pd.Series) : A pandas Series that contains the target labels.    
    Returns:
    Dict[str,int] : The created dictionary which maps the unique targets to consecutive integers.
    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0
    
    print(f"There are {len(fam2label)} labels.")
        
    return fam2label

def get_amino_acid_frequencies(data: List[str])->pd.DataFrame:
    """
    Accepts a list of protein sequences and calculate the frequency of each amino acid in it
    
    Parameters:
    data (List[str]) : A list of protein sequences
    
    Returns:
    pd.DataFrame : A dataframe with two columns 'AA' and 'Frequency' showing the amino acid and it's frequency respectively.
    """
    aa_counter = Counter()
    
    for sequence in data:
        aa_counter.update(sequence)
        
    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})

def build_vocab(data, rare_AA_count = 4)->Dict[str,int]:
    """
    Builds a vocabulary of amino acids from a list of sequences and creates a mapping from AA strings to unique integers.
    
    Parameters:
        - data (list): a list of sequences, where each sequence is a string of amino acids.
        - rare_AA_count (int, optional): The number of rare AAs to remove from the vocabulary. Defaults to 4.
        
    Returns:
        - word2id (dict): a dictionary that maps each unique AA in data to a unique integer. The integers are assigned in the order that the AAs appear in the data, starting from 2.
        Two additional special tokens, '<pad>' and '<unk>', are added to the mapping, they are assigned the integers 0 and 1, respectively.
    """
    # Build the vocabulary
    voc = dict()
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

