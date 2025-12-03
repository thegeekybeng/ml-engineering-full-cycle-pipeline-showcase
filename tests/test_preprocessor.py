"""
Unit tests for Preprocessor module.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    """Test cases for Preprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config._config['output']['verbose'] = False
        self.preprocessor = Preprocessor(self.config)
        
        # Create sample data
        self.X = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5] * 20,
            'numeric2': [1.5, 2.5, 3.5, 4.5, 5.5] * 20,
            'categorical': ['A', 'B', 'C', 'D', 'E'] * 20
        })
        self.y = pd.Series([0, 1, 0, 1, 0] * 20)
    
    def test_init(self):
        """Test Preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor.config)
    
    def test_split_data(self):
        """Test train-test split."""
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(self.X, self.y)
        
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
    
    def test_create_pipeline(self):
        """Test preprocessing pipeline creation."""
        numerical_cols = ['numeric1', 'numeric2']
        categorical_cols = ['categorical']
        
        pipeline = self.preprocessor.create_pipeline(numerical_cols, categorical_cols)
        self.assertIsNotNone(pipeline)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        numerical_cols = ['numeric1', 'numeric2']
        categorical_cols = ['categorical']
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(self.X, self.y)
        X_train_transformed = self.preprocessor.fit_transform(
            X_train, numerical_cols, categorical_cols
        )
        
        self.assertIsNotNone(X_train_transformed)
        self.assertGreater(X_train_transformed.shape[1], X_train.shape[1])


if __name__ == '__main__':
    unittest.main()

