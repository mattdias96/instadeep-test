import unittest
import pandas as pd
import torch

from data.dataset import SequenceDataset, reader, get_amino_acid_frequencies
    
    
class TestData(unittest.TestCase):    
    """
    A class of unit tests to test the functionality of the functions related to data preparation
    """
    def testPreprocess(self):
        """
        A test for the preprocess function of the SequenceDataset class.
        This test asserts that the output of the function is as expected when given a specific input.
        """
        self.max_len = 10
        self.word2id = {'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 
                        'K': 11, 'L': 12, 'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 
                        'T': 19, 'V': 20, 'W': 21, 'Y': 22, '<pad>': 0, '<unk>': 1}
        text = 'ABCDEFGHIJ'
        expected_output = torch.tensor(
            [[[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]],
            dtype=torch.float32)
        self.assertTrue(torch.allclose(SequenceDataset.preprocess(self, text), expected_output, rtol=1e-5, atol=1e-5))

    def testReaderFunction(self, dir_path):
        """
        A test for the test reader function.
        """
        train_data, train_targets = reader("train", dir_path)
        self.assertIsInstance(train_data, pd.Series)
        self.assertIsInstance(train_targets, pd.Series)
        self.assertGreater(len(train_data), 0)
        self.assertEqual(len(train_data), len(train_targets))

    
    def testGetAminoAcidFrequencies(self):
        """
        A test for the get_amino_acid_frequencies function
        """
        data = ['ABCDEFGHIJ', 'KLMNOPQRST']
        expected_output = pd.DataFrame({'AA': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'],
                                         'Frequency': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                         })
        self.assertTrue(expected_output.equals(get_amino_acid_frequencies(data)))