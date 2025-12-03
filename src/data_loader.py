"""
Data Loading Module

Handles loading data from SQLite database, including:
- Downloading database from URL if needed
- Connecting to SQLite database
- Loading data into pandas DataFrame
- Separating features and target variable
- Handling missing values (median imputation + indicator variable)
- Identifying feature types (numerical vs categorical)
"""

import os
import sqlite3
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from src.config import Config


class DataLoader:
    """
    Data loader for SQLite database.
    
    Handles downloading, loading, and initial preprocessing of the dataset.
    """
    
    def __init__(self, config: Config):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.db_url = config.get('data', 'db_url')
        self.temp_db_path = config.get('data', 'temp_db_path', default='data/phishing.db')
        self.verbose = config.get('output', 'verbose', default=True)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.temp_db_path) if os.path.dirname(self.temp_db_path) else 'data', exist_ok=True)
    
    def download_database(self) -> None:
        """Download database file from URL if it doesn't exist."""
        if os.path.exists(self.temp_db_path):
            if self.verbose:
                print("✓ Using existing database file")
            return
        
        if self.verbose:
            print("📥 Downloading database from URL...")
        
        try:
            urllib.request.urlretrieve(self.db_url, self.temp_db_path)
            if self.verbose:
                print("✓ Database downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download database from {self.db_url}: {e}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from SQLite database.
        
        Returns:
            DataFrame containing the dataset
        """
        if self.verbose:
            print("=" * 70)
            print("STEP 2: Loading Dataset")
            print("=" * 70)
        
        # Download database if needed
        self.download_database()
        
        # Connect to SQLite database
        try:
            conn = sqlite3.connect(self.temp_db_path)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database {self.temp_db_path}: {e}")
        
        # Get table name
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_result = cursor.fetchone()
        
        if not table_result:
            conn.close()
            raise RuntimeError("No tables found in database")
        
        table_name = table_result[0]
        
        if self.verbose:
            print(f"📊 Table name: {table_name}")
        
        # Load data
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            conn.close()
            raise RuntimeError(f"Failed to load data from table {table_name}: {e}")
        finally:
            conn.close()
        
        # Remove index column if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            if self.verbose:
                print("✓ Removed index column (Unnamed: 0)")
        
        if self.verbose:
            print(f"\n📈 Dataset Summary:")
            print(f"   • Shape: {df.shape[0]:,} samples × {df.shape[1]} features")
            print(f"   • Features: {df.shape[1] - 1}")
            print("=" * 70)
        
        return df
    
    def separate_features_target(self, df: pd.DataFrame, target_column: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
        
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target Series
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()
        
        if self.verbose:
            print("=" * 70)
            print("STEP 3: Feature and Target Separation")
            print("=" * 70)
            print(f"\n📊 Features Summary:")
            print(f"   • Shape: {X.shape[0]:,} samples × {X.shape[1]} features")
            print(f"   • Feature columns: {X.shape[1]}")
            print("=" * 70)
        
        return X, y
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using median imputation and create indicator variable.
        
        Based on EDA findings:
        - LineOfCode has 22.43% missing values
        - Create LineOfCode_Missing indicator variable
        
        Args:
            X: Features DataFrame
        
        Returns:
            DataFrame with missing values handled
        """
        X_processed = X.copy()
        
        # Check for missing values
        missing_cols = X_processed.columns[X_processed.isnull().any()].tolist()
        
        if not missing_cols:
            if self.verbose:
                print("✓ No missing values found")
            return X_processed
        
        if self.verbose:
            print("=" * 70)
            print("HANDLING MISSING VALUES")
            print("=" * 70)
            print(f"\n📊 Columns with missing values: {len(missing_cols)}")
            for col in missing_cols:
                missing_count = X_processed[col].isnull().sum()
                missing_pct = (missing_count / len(X_processed)) * 100
                print(f"   • {col}: {missing_count:,} ({missing_pct:.2f}%)")
        
        # Handle LineOfCode missing values (as per EDA findings)
        if 'LineOfCode' in missing_cols:
            # Create indicator variable before imputation
            X_processed['LineOfCode_Missing'] = X_processed['LineOfCode'].isnull().astype(int)
            
            # Median imputation
            median_value = X_processed['LineOfCode'].median()
            X_processed = X_processed.assign(LineOfCode=X_processed['LineOfCode'].fillna(median_value))
            
            if self.verbose:
                print(f"\n✓ Created LineOfCode_Missing indicator variable")
                print(f"✓ Imputed LineOfCode with median: {median_value:.2f}")
        
        # Handle any other missing values with median (for numerical) or mode (for categorical)
        for col in missing_cols:
            if col == 'LineOfCode':
                continue  # Already handled
            
            if X_processed[col].dtype in ['int64', 'float64']:
                # Numerical: use median
                median_value = X_processed[col].median()
                X_processed = X_processed.assign(**{col: X_processed[col].fillna(median_value)})
                if self.verbose:
                    print(f"✓ Imputed {col} (numerical) with median: {median_value:.2f}")
            else:
                # Categorical: use mode
                mode_value = X_processed[col].mode()[0] if not X_processed[col].mode().empty else 'Unknown'
                X_processed = X_processed.assign(**{col: X_processed[col].fillna(mode_value)})
                if self.verbose:
                    print(f"✓ Imputed {col} (categorical) with mode: {mode_value}")
        
        if self.verbose:
            print("\n✓ Missing value handling complete")
            print("=" * 70)
        
        return X_processed
    
    def identify_feature_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify numerical and categorical features.
        
        Args:
            X: Features DataFrame
        
        Returns:
            Dictionary with 'numerical' and 'categorical' keys containing feature lists
            
        Raises:
            ValueError: If DataFrame is empty
        """
        if X.empty:
            raise ValueError("DataFrame is empty - cannot identify feature types")
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.verbose:
            print("=" * 70)
            print("FEATURE TYPE IDENTIFICATION")
            print("=" * 70)
            
            print(f"\n📊 Categorical Features ({len(categorical_cols)}):")
            for col in categorical_cols:
                print(f"   • {col}")
            
            print(f"\n📊 Numerical Features ({len(numerical_cols)}):")
            for col in numerical_cols:
                print(f"   • {col}")
            
            # Check categorical feature cardinality
            if categorical_cols:
                print(f"\n📈 Categorical Feature Cardinality:")
                for col in categorical_cols:
                    unique_count = X[col].nunique()
                    top_5 = X[col].value_counts().head().to_dict()
                    print(f"\n   {col}:")
                    print(f"      • Unique values: {unique_count}")
                    print(f"      • Top 5 categories: {list(top_5.keys())[:5]}")
            
            print("=" * 70)
        
        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
    
    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
        """
        Complete data loading and preparation pipeline.
        
        Returns:
            Tuple of (X, y, feature_types) where:
            - X: Features DataFrame (with missing values handled)
            - y: Target Series
            - feature_types: Dictionary with 'numerical' and 'categorical' feature lists
        """
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Separate features and target
        X, y = self.separate_features_target(df)
        
        # Step 3: Handle missing values
        X = self.handle_missing_values(X)
        
        # Step 4: Identify feature types
        feature_types = self.identify_feature_types(X)
        
        return X, y, feature_types


if __name__ == '__main__':
    # Test data loader
    from src.config import Config
    
    config = Config()
    loader = DataLoader(config)
    
    try:
        X, y, feature_types = loader.load_and_prepare()
        print(f"\n✅ Data loading completed successfully!")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Numerical features: {len(feature_types['numerical'])}")
        print(f"   Categorical features: {len(feature_types['categorical'])}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

