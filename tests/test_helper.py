"""
This module contains unit tests
for methods available in the helper
file
"""
import unittest
import pandas as pd
import torch

from utils import buildVocab, buildLabels, loadData

class TestHelper(unittest.TestCase):
    """
    A class of unit tests to test the functionality of the helper functions
    """
    def testBuildVocabFunction(self):
        """
        A test to test the functionality of the buildVocab function
        """
        # Test that the buildVocab function correctly builds a vocabulary
        # from AA strings to unique integers
        train_data = ["AAA", "AAC", "AGG"]
        word2id = buildVocab(train_data, 1)
        self.assertIsInstance(word2id, dict)
        self.assertEqual(len(word2id), 4)
        self.assertEqual(word2id["A"], 2)
        self.assertFalse("C" in word2id)
        self.assertEqual(word2id["G"], 3)

    def testBuildLabels(self):
        """
        A test to test the functionality of the buildLabels function
        """
        targets = pd.Series(['a', 'b', 'c', 'a'])
        expected_output = {'a': 1, 'b': 2, 'c': 3, '<unk>': 0}
        self.assertEqual(buildLabels(targets), expected_output)
        targets = pd.Series([], dtype = "int64")
        expected_output = {'<unk>': 0}
        self.assertEqual(buildLabels(targets), expected_output)
        targets = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
        expected_output = {'a': 1, 'b': 2, 'c': 3, '<unk>': 0}
        self.assertEqual(buildLabels(targets), expected_output)

    def testLoadData(self):
        """
        A test to test the functionality of the loadData function
        """
        num_workers = 4
        word2id = {'A': 1, 'B': 2, 'C': 3}
        fam2label = {'Family1': 1, 'Family2': 2} # change this
        seq_max_len = 10
        data_dir = "/Users/mathe/Documents/random_split" # change this
        batch_size = 8

        dataloaders = loadData(num_workers, word2id, fam2label,
                               seq_max_len, data_dir, batch_size)

        self.assertIsInstance(dataloaders, dict)
        self.assertIsInstance(dataloaders['train'], torch.utils.data.DataLoader)
        self.assertIsInstance(dataloaders['dev'], torch.utils.data.DataLoader)
        self.assertIsInstance(dataloaders['test'], torch.utils.data.DataLoader)
        self.assertEqual(dataloaders['train'].batch_size, batch_size)
        self.assertEqual(dataloaders['dev'].batch_size, batch_size)
        self.assertEqual(dataloaders['test'].batch_size, batch_size)

    # Add good tests for evaluate and predict if have time
