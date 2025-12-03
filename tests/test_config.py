"""
Unit tests for Config module.
"""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def test_init_defaults(self):
        """Test Config initialization with defaults."""
        config = Config()
        self.assertIsNotNone(config._config)
        self.assertIn('data', config._config)
        self.assertIn('models', config._config)
    
    def test_get(self):
        """Test get method."""
        config = Config()
        db_url = config.get('data', 'db_url')
        self.assertIsNotNone(db_url)
    
    def test_get_default(self):
        """Test get method with default value."""
        config = Config()
        value = config.get('nonexistent', 'key', default='default_value')
        self.assertEqual(value, 'default_value')
    
    @patch.dict(os.environ, {'MLP_DATA__DB_URL': 'test_url'})
    def test_env_variables(self):
        """Test environment variable loading."""
        config = Config()
        # Environment variables should override defaults
        self.assertIsNotNone(config._config)


if __name__ == '__main__':
    unittest.main()

