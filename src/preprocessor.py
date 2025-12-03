"""
Preprocessing Module

Handles data preprocessing pipeline including:
- RobustScaler for numerical features (robust to outliers)
- OneHotEncoder for categorical features
- ColumnTransformer pipeline
- Train-test split with stratification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
from src.config import Config


class Preprocessor:
    """
    Preprocessing pipeline for MLP.
    
    Applies RobustScaler to numerical features and OneHotEncoder to categorical features.
    Based on EDA findings: RobustScaler recommended due to right-skewed distributions and outliers.
    """
    
    def __init__(self, config: Config):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.verbose = config.get('output', 'verbose', default=True)
        
        # Get preprocessing settings from config
        self.scaler_type = config.get('preprocessing', 'scaler', default='RobustScaler')
        self.onehot_drop = config.get('preprocessing', 'onehot_drop', default='first')
        self.onehot_handle_unknown = config.get('preprocessing', 'onehot_handle_unknown', default='ignore')
        
        self.preprocessor = None
        self.numerical_cols = None
        self.categorical_cols = None
    
    def create_pipeline(self, numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        """
        Create preprocessing pipeline using ColumnTransformer.
        
        Args:
            numerical_cols: List of numerical feature column names
            categorical_cols: List of categorical feature column names
        
        Returns:
            ColumnTransformer pipeline
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Create scaler based on config
        if self.scaler_type == 'RobustScaler':
            scaler = RobustScaler()
        elif self.scaler_type == 'StandardScaler':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}. Use 'RobustScaler' or 'StandardScaler'")
        
        # Create preprocessing pipeline
        transformers = []
        
        # Numerical features: RobustScaler (robust to outliers, as per EDA findings)
        if numerical_cols:
            transformers.append(('num', scaler, numerical_cols))
        
        # Categorical features: OneHotEncoder
        if categorical_cols:
            transformers.append((
                'cat',
                OneHotEncoder(
                    drop=self.onehot_drop,
                    sparse_output=False,
                    handle_unknown=self.onehot_handle_unknown
                ),
                categorical_cols
            ))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        if self.verbose:
            print("=" * 70)
            print("PREPROCESSING PIPELINE CREATED")
            print("=" * 70)
            print("✓ Pipeline Components:")
            if numerical_cols:
                print(f"   • Numerical features ({len(numerical_cols)}): {self.scaler_type} (robust to outliers)")
            if categorical_cols:
                print(f"   • Categorical features ({len(categorical_cols)}): OneHotEncoder (drop='{self.onehot_drop}')")
            print("=" * 70)
        
        return self.preprocessor
    
    def fit_transform(self, X_train: pd.DataFrame, numerical_cols: List[str], 
                     categorical_cols: List[str]) -> np.ndarray:
        """
        Fit preprocessing pipeline on training data and transform it.
        
        Args:
            X_train: Training features DataFrame
            numerical_cols: List of numerical feature column names
            categorical_cols: List of categorical feature column names
        
        Returns:
            Transformed training data as numpy array
        """
        if self.preprocessor is None:
            self.create_pipeline(numerical_cols, categorical_cols)
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        if self.verbose:
            print(f"\n✓ Preprocessing applied to training data")
            print(f"   Original shape: {X_train.shape}")
            print(f"   Transformed shape: {X_train_transformed.shape}")
        
        return X_train_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessing pipeline.
        
        Args:
            X: Features DataFrame
        
        Returns:
            Transformed data as numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        X_transformed = self.preprocessor.transform(X)
        
        if self.verbose:
            print(f"✓ Preprocessing applied to data")
            print(f"   Original shape: {X.shape}")
            print(f"   Transformed shape: {X_transformed.shape}")
        
        return X_transformed
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets with stratification.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If inputs are empty or have mismatched lengths
        """
        if X.empty or y.empty:
            raise ValueError("Input data is empty - cannot split")
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")
        """
        Split data into training and testing sets with stratification.
        
        Args:
            X: Features DataFrame
            y: Target Series
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.config.get('data', 'test_size', default=0.2)
        random_state = self.config.get('data', 'random_state', default=42)
        stratify = self.config.get('data', 'stratify', default=True)
        
        if self.verbose:
            print("=" * 70)
            print("STEP 4: Train-Test Split")
            print("=" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        train_pct = X_train.shape[0] / len(X) * 100
        test_pct = X_test.shape[0] / len(X) * 100
        
        if self.verbose:
            print(f"\n📊 Dataset Split:")
            print(f"   • Training set: {X_train.shape[0]:,} samples ({train_pct:.1f}%)")
            print(f"   • Test set: {X_test.shape[0]:,} samples ({test_pct:.1f}%)")
            
            print(f"\n📈 Training Set Class Distribution:")
            train_counts = y_train.value_counts()
            train_pct_dist = y_train.value_counts(normalize=True) * 100
            for label, count in train_counts.items():
                label_name = "Legitimate" if label == 1 else "Phishing"
                print(f"   • {label_name} ({label}): {count:,} ({train_pct_dist[label]:.2f}%)")
            
            print(f"\n📈 Test Set Class Distribution:")
            test_counts = y_test.value_counts()
            test_pct_dist = y_test.value_counts(normalize=True) * 100
            for label, count in test_counts.items():
                label_name = "Legitimate" if label == 1 else "Phishing"
                print(f"   • {label_name} ({label}): {count:,} ({test_pct_dist[label]:.2f}%)")
            
            print("\n✓ Data split complete with stratification")
            print("=" * 70)
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing transformation.
        
        Returns:
            List of feature names after one-hot encoding
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before getting feature names.")
        
        # Get feature names from ColumnTransformer
        feature_names = []
        
        # Numerical features (same names)
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        
        # Categorical features (expanded by OneHotEncoder)
        if self.categorical_cols:
            # Get one-hot encoded feature names
            for col in self.categorical_cols:
                # Get categories from the fitted OneHotEncoder
                cat_transformer = None
                for name, transformer, cols in self.preprocessor.transformers_:
                    if name == 'cat' and col in cols:
                        cat_transformer = transformer
                        break
                
                if cat_transformer is not None:
                    categories = cat_transformer.categories_[self.categorical_cols.index(col)]
                    # Drop first category if drop='first'
                    if self.onehot_drop == 'first' and len(categories) > 1:
                        categories = categories[1:]
                    # Add feature names
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
        
        return feature_names


if __name__ == '__main__':
    # Test preprocessor
    from src.config import Config
    from src.data_loader import DataLoader
    
    config = Config()
    
    # Load data
    loader = DataLoader(config)
    X, y, feature_types = loader.load_and_prepare()
    
    # Create preprocessor
    preprocessor = Preprocessor(config)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Fit and transform
    X_train_transformed = preprocessor.fit_transform(
        X_train,
        feature_types['numerical'],
        feature_types['categorical']
    )
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"\n✅ Preprocessing completed successfully!")
    print(f"   Training shape: {X_train_transformed.shape}")
    print(f"   Test shape: {X_test_transformed.shape}")

