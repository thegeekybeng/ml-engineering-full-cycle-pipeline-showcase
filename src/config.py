"""
Configuration Management Module

Handles loading configuration from multiple sources:
1. YAML/JSON config file (primary)
2. Environment variables (override)
3. Command-line arguments (highest priority)
4. Default values (fallback)

This module enables easy experimentation with different algorithms and parameters.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config:
    """
    Configuration manager for the MLP pipeline.
    
    Supports multiple configuration sources with priority order:
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. Config file (YAML/JSON)
    4. Default values (lowest priority)
    """
    
    def __init__(self, config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config file (YAML or JSON)
            args: Command-line arguments (optional)
        """
        # Load defaults first
        self._config = self._get_defaults()
        
        # Load from config file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Override with environment variables
        self.load_from_env()
        
        # Override with command-line arguments (highest priority)
        if args:
            self.load_from_args(args)
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'data': {
                'db_url': 'https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db',
                'temp_db_path': 'data/phishing.db',
                'test_size': 0.2,
                'random_state': 42,
                'stratify': True
            },
            'preprocessing': {
                'scaler': 'RobustScaler',
                'handle_missing': 'median_imputation',
                'create_indicator': True,
                'onehot_drop': 'first',
                'onehot_handle_unknown': 'ignore'
            },
            'models': {
                'enabled': [
                    'LogisticRegression',
                    'RandomForest',
                    'GradientBoosting',
                    'XGBoost'
                ],
                'LogisticRegression': {
                    'max_iter': 1000,
                    'random_state': 42,
                    'solver': 'lbfgs'
                },
                'RandomForest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'GradientBoosting': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                },
                'SVM': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'random_state': 42,
                    'probability': True
                },
                'KNN': {
                    'n_neighbors': 5,
                    'weights': 'uniform'
                },
                'NaiveBayes': {},
                'NeuralNetwork': {
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 500,
                    'learning_rate': 'constant',
                    'learning_rate_init': 0.001,
                    'random_state': 42
                },
                'XGBoost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                },
                'LightGBM': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'RandomizedSearchCV',
                'n_iter': 50,
                'cv': 5,
                'n_jobs': -1,
                'random_state': 42
            },
            'cross_validation': {
                'enabled': True,
                'cv_folds': 5,
                'scoring': ['accuracy', 'roc_auc', 'f1']
            },
            'evaluation': {
                'metrics': [
                    'accuracy',
                    'precision',
                    'recall',
                    'f1',
                    'specificity',
                    'fpr',
                    'fnr',
                    'balanced_accuracy',
                    'mcc',
                    'roc_auc',
                    'pr_auc'
                ]
            },
            'output': {
                'results_dir': 'results',
                'save_models': False,
                'save_predictions': False,
                'verbose': True
            }
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                if not YAML_AVAILABLE:
                    raise ImportError("YAML support requires PyYAML. Install with: pip install pyyaml")
                file_config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}. Use .yaml, .yml, or .json")
        
        # Deep merge with existing config
        self._config = self._deep_merge(self._config, file_config)
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'MLP_DB_URL': ('data', 'db_url'),
            'MLP_TEST_SIZE': ('data', 'test_size', float),
            'MLP_RANDOM_STATE': ('data', 'random_state', int),
            'MLP_SCALER': ('preprocessing', 'scaler'),
            'MLP_MODELS': ('models', 'enabled', lambda x: x.split(',')),
            'MLP_RESULTS_DIR': ('output', 'results_dir'),
            'MLP_VERBOSE': ('output', 'verbose', lambda x: x.lower() == 'true')
        }
        
        for env_var, (section, key, *converters) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                converter = converters[0] if converters else lambda x: x
                try:
                    self._config[section][key] = converter(value)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not set {env_var}: {e}")
    
    def load_from_args(self, args: argparse.Namespace):
        """Load configuration from command-line arguments."""
        if hasattr(args, 'db_url') and args.db_url:
            self._config['data']['db_url'] = args.db_url
        
        if hasattr(args, 'test_size') and args.test_size:
            self._config['data']['test_size'] = args.test_size
        
        if hasattr(args, 'random_state') and args.random_state:
            self._config['data']['random_state'] = args.random_state
        
        if hasattr(args, 'model') and args.model:
            # If specific model requested, enable only that model
            if args.model != 'all':
                self._config['models']['enabled'] = [args.model]
        
        if hasattr(args, 'results_dir') and args.results_dir:
            self._config['output']['results_dir'] = args.results_dir
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, *keys, default=None):
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('data', 'db_url')
            config.get('models', 'RandomForest', 'n_estimators')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self._config[key]
    
    def __contains__(self, key):
        """Check if key exists."""
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Machine Learning Pipeline (MLP) for Phishing Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help='Database URL (overrides config file)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=None,
        help='Test set size (overrides config file)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help='Random seed (overrides config file)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost'],
        default='all',
        help='Model to train (default: all)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory (overrides config file)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file
        args: Command-line arguments
    
    Returns:
        Config object
    """
    return Config(config_path=config_path, args=args)


if __name__ == '__main__':
    # Test configuration loading
    parser = create_parser()
    args = parser.parse_args()
    
    config = load_config(config_path=args.config, args=args)
    
    print("=" * 70)
    print("CONFIGURATION LOADED")
    print("=" * 70)
    print(f"Database URL: {config.get('data', 'db_url')}")
    print(f"Test Size: {config.get('data', 'test_size')}")
    print(f"Random State: {config.get('data', 'random_state')}")
    print(f"Enabled Models: {config.get('models', 'enabled')}")
    print("=" * 70)

