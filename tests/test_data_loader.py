"""
Unit tests for DataLoader module.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config._config['output']['verbose'] = False
        self.loader = DataLoader(self.config)
    
    def test_init(self):
        """Test DataLoader initialization."""
        self.assertIsNotNone(self.loader.config)
        self.assertIsNotNone(self.loader.db_url)
    
    @patch('src.data_loader.os.path.exists')
    @patch('src.data_loader.urllib.request.urlretrieve')
    def test_download_database_new(self, mock_urlretrieve, mock_exists):
        """Test database download when file doesn't exist."""
        mock_exists.return_value = False
        self.loader.download_database()
        mock_urlretrieve.assert_called_once()
    
    @patch('src.data_loader.os.path.exists')
    def test_download_database_exists(self, mock_exists):
        """Test database download when file exists."""
        mock_exists.return_value = True
        # Should not raise error
        self.loader.download_database()
    
    def test_handle_missing_values_no_missing(self):
        """Test missing value handling with no missing values."""
        df = pd.DataFrame({
            'LineOfCode': [100, 200, 300],
            'OtherCol': [1, 2, 3]
        })
        result = self.loader.handle_missing_values(df)
        self.assertEqual(len(result), len(df))
        self.assertIn('LineOfCode_Missing', result.columns)
    
    def test_handle_missing_values_with_missing(self):
        """Test missing value handling with missing values."""
        df = pd.DataFrame({
            'LineOfCode': [100, np.nan, 300],
            'OtherCol': [1, 2, 3]
        })
        result = self.loader.handle_missing_values(df)
        self.assertFalse(result['LineOfCode'].isnull().any())
        self.assertIn('LineOfCode_Missing', result.columns)
        self.assertEqual(result['LineOfCode_Missing'].sum(), 1)
    
    def test_identify_feature_types(self):
        """Test feature type identification."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [1.5, 2.5, 3.5],
            'categorical': ['A', 'B', 'C']
        })
        numerical, categorical = self.loader.identify_feature_types(df)
        self.assertEqual(len(numerical), 2)
        self.assertEqual(len(categorical), 1)
        self.assertIn('numeric1', numerical)
        self.assertIn('categorical', categorical)


if __name__ == '__main__':
    unittest.main()

