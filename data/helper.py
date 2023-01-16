"""
This module contains helper functions
for the SequenceDataset class
"""
import os

from collections import Counter
import pandas as pd


def reader(partition:str, data_path:str)->tuple:
    """
    Reads files from a specified partition within a given data path, loads the contents
    of those files into a Pandas DataFrame, concatenates all the dataframe into single one,
    and finally returns 2 specific columns of the dataframe : 'sequence' (regressors) and
    'family_accession' (target).

    Parameters:
    partition (str) : The name of the partition to read files from
    data_path (str) : The path to the data

    Returns:
    Tuple : (pandas.Series, pandas.Series) - containing the 'sequence' and 'family_accession'
    columns of the concatenated DataFrame
    """
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name), encoding="utf-8") as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)

    return all_data["sequence"], all_data["family_accession"]

def get_amino_acid_frequencies(data: list)->pd.DataFrame:
    """
    Accepts a list of protein sequences and calculate the frequency of each amino acid in it

    Parameters:
    data (List[str]) : A list of protein sequences

    Returns:
    pd.DataFrame : A dataframe with two columns 'AA' and 'Frequency' showing the amino acid and
    it's frequency respectively.
    """
    aa_counter = Counter()

    for sequence in data:
        aa_counter.update(sequence)

    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})

