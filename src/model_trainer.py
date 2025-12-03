"""
Model Training Module

Handles training of all machine learning models including:
- Streamlined model set (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- Hyperparameter configuration from config
- Hyperparameter tuning via RandomizedSearchCV (optional)
- Training progress tracking

Note: The pipeline uses a streamlined 4-model configuration for optimal balance
of performance and computational efficiency. Additional models (SVM, KNN, Naive Bayes,
Neural Network, LightGBM) were evaluated but excluded from the final pipeline.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from src.config import Config

# Optional advanced models (XGBoost, LightGBM)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """
    Model trainer for MLP pipeline.
    
    Creates and trains multiple machine learning models based on configuration.
    Default configuration uses 4 streamlined models: Logistic Regression, Random Forest,
    Gradient Boosting, and XGBoost for optimal performance-efficiency balance.
    """
    
    def __init__(self, config: Config):
        """
        Initialize model trainer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.verbose = config.get('output', 'verbose', default=True)
        self.enabled_models = config.get('models', 'enabled', default=[])
        self.trained_models = {}
        self.training_info = {}
    
    def _create_model(self, model_name: str):
        """
        Create a model instance based on name and configuration.
        
        Args:
            model_name: Name of the model to create
        
        Returns:
            Model instance
        """
        model_config = self.config.get('models', model_name, default={})
        random_state = self.config.get('data', 'random_state', default=42)
        
        if model_name == 'LogisticRegression':
            return LogisticRegression(
                max_iter=model_config.get('max_iter', 1000),
                random_state=random_state,
                solver=model_config.get('solver', 'lbfgs'),
                C=model_config.get('C', 1.0)
            )
        
        elif model_name == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', 10),
                random_state=random_state,
                n_jobs=model_config.get('n_jobs', -1)
            )
        
        elif model_name == 'GradientBoosting':
            return GradientBoostingClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                learning_rate=model_config.get('learning_rate', 0.1),
                max_depth=model_config.get('max_depth', 5),
                random_state=random_state
            )
        
        elif model_name == 'SVM':
            return SVC(
                kernel=model_config.get('kernel', 'rbf'),
                C=model_config.get('C', 1.0),
                gamma=model_config.get('gamma', 'scale'),
                random_state=random_state,
                probability=model_config.get('probability', True)
            )
        
        elif model_name == 'KNN':
            return KNeighborsClassifier(
                n_neighbors=model_config.get('n_neighbors', 5),
                weights=model_config.get('weights', 'uniform')
            )
        
        elif model_name == 'NaiveBayes':
            return GaussianNB()
        
        elif model_name == 'NeuralNetwork':
            return MLPClassifier(
                hidden_layer_sizes=tuple(model_config.get('hidden_layer_sizes', [100, 50])),
                max_iter=model_config.get('max_iter', 500),
                learning_rate=model_config.get('learning_rate', 'constant'),
                learning_rate_init=model_config.get('learning_rate_init', 0.001),
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=10
            )
        
        elif model_name == 'XGBoost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            return xgb.XGBClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                learning_rate=model_config.get('learning_rate', 0.1),
                max_depth=model_config.get('max_depth', 5),
                random_state=random_state,
                eval_metric='logloss'
            )
        
        elif model_name == 'LightGBM':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            return lgb.LGBMClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                learning_rate=model_config.get('learning_rate', 0.1),
                max_depth=model_config.get('max_depth', 5),
                random_state=random_state,
                verbose=-1
            )
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def _get_display_name(self, model_name: str) -> str:
        """Convert config model name to display name."""
        name_mapping = {
            'LogisticRegression': 'Logistic Regression',
            'RandomForest': 'Random Forest',
            'GradientBoosting': 'Gradient Boosting',
            'SVM': 'SVM (RBF)',
            'KNN': 'K-Nearest Neighbors',
            'NaiveBayes': 'Naive Bayes',
            'NeuralNetwork': 'Neural Network (MLP)',
            'XGBoost': 'XGBoost',
            'LightGBM': 'LightGBM'
        }
        return name_mapping.get(model_name, model_name)
    
    def _get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameter search space for a model.
        
        Args:
            model_name: Name of the model (config name, not display name)
        
        Returns:
            Dictionary of hyperparameter distributions for RandomizedSearchCV
        """
        random_state = self.config.get('data', 'random_state', default=42)
        
        if model_name == 'LogisticRegression':
            return {
                'C': loguniform(1e-3, 1e2),
                'solver': ['lbfgs', 'liblinear'],  # Removed 'sag' - causes convergence warnings
                'max_iter': [1000, 2000, 3000]  # Increased to ensure convergence
            }
        
        elif model_name == 'RandomForest':
            return {
                'n_estimators': randint(50, 300),
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2', None]
            }
        
        elif model_name == 'GradientBoosting':
            return {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5)
            }
        
        elif model_name == 'SVM':
            return {
                'C': loguniform(1e-2, 1e2),
                'gamma': ['scale', 'auto', loguniform(1e-4, 1e-1)],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        elif model_name == 'KNN':
            return {
                'n_neighbors': randint(3, 20),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        
        elif model_name == 'NeuralNetwork':
            return {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100), (100, 50, 25)],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': loguniform(1e-4, 1e-1),
                'alpha': loguniform(1e-5, 1e-1),
                'max_iter': [300, 500, 700]
            }
        
        elif model_name == 'XGBoost':
            # Focused search space - most impactful hyperparameters only
            # With 9 hyperparameters, n_iter=50 is too sparse. Reduced to 5 key params.
            return {
                'n_estimators': randint(100, 300),  # More trees often better
                'learning_rate': uniform(0.05, 0.15),  # Focus around good defaults (0.1)
                'max_depth': randint(3, 7),  # Range 3-6, centered around default (5)
                'subsample': uniform(0.8, 0.2),  # Range 0.8-1.0 (moderate subsampling)
                'colsample_bytree': uniform(0.8, 0.2)  # Range 0.8-1.0 (moderate feature sampling)
                # Removed: min_child_weight, gamma, reg_alpha, reg_lambda
                # These have smaller impact and expand search space too much for n_iter=50
            }
        
        elif model_name == 'LightGBM':
            return {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'num_leaves': randint(20, 100),
                'min_child_samples': randint(10, 50),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        
        # NaiveBayes has no hyperparameters to tune
        return {}
    
    def _tune_hyperparameters(self, model_name: str, model: Any, 
                              X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            model_name: Name of the model (config name)
            model: Base model instance
            X_train: Training features
            y_train: Training target
        
        Returns:
            Best model from hyperparameter search
        """
        # Get hyperparameter search space
        param_distributions = self._get_hyperparameter_space(model_name)
        
        # If no hyperparameters to tune, return model as-is
        if not param_distributions:
            return model
        
        # Get hyperparameter tuning config
        tuning_config = self.config.get('hyperparameter_tuning', default={})
        n_iter = tuning_config.get('n_iter', 50)
        cv = tuning_config.get('cv', 5)
        n_jobs = tuning_config.get('n_jobs', -1)
        random_state = tuning_config.get('random_state', 42)
        scoring = 'accuracy'  # Use accuracy as primary metric
        
        if self.verbose:
            print(f"   🔍 Performing hyperparameter tuning (n_iter={n_iter}, cv={cv})...")
        
        # Suppress convergence warnings during hyperparameter tuning (they're expected during search)
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=0,
                return_train_score=True
            )
            
            random_search.fit(X_train, y_train)
        
        if self.verbose:
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            print(f"   ✓ Best CV score: {best_score:.4f}")
            print(f"   ✓ Best parameters: {best_params}")
        
        return random_search.best_estimator_
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create all enabled models based on configuration.
        
        Returns:
            Dictionary of model name -> model instance
        """
        models = {}
        
        # If no specific models enabled, use streamlined 4-model set
        if not self.enabled_models or 'all' in self.enabled_models:
            self.enabled_models = [
                'LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost'
            ]
            # Note: Removed SVM, KNN, NaiveBayes, NeuralNetwork, LightGBM for optimal runtime
            # These models are still supported if explicitly enabled in config, but not in default set
        
        if self.verbose:
            print("=" * 70)
            print("STEP 5: MODEL DEFINITIONS")
            print("=" * 70)
            print(f"\n📊 Creating {len(self.enabled_models)} models...")
        
        for model_name in self.enabled_models:
            try:
                display_name = self._get_display_name(model_name)
                model = self._create_model(model_name)
                models[display_name] = model
                
                if self.verbose:
                    print(f"   ✓ {display_name}: {type(model).__name__}")
            
            except ImportError as e:
                if self.verbose:
                    print(f"   ⚠️  {model_name}: Not available ({e})")
            except Exception as e:
                if self.verbose:
                    print(f"   ✗ {model_name}: Failed to create ({e})")
        
        if self.verbose:
            print("=" * 70)
        
        return models
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, 
                  models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train all models on training data.
        
        Args:
            X_train: Training features (preprocessed)
            y_train: Training target
            models: Dictionary of models to train (if None, creates from config)
        
        Returns:
            Dictionary of trained models
        """
        if models is None:
            models = self.create_models()
        
        if self.verbose:
            print("=" * 70)
            print("TRAINING MACHINE LEARNING MODELS")
            print("=" * 70)
            
            print("\n📚 Training Process Explanation:")
            print("-" * 70)
            print("""
Different models use different training approaches:

1. **Logistic Regression**:
   • Uses iterations (max_iter=1000) until convergence
   • Automatically stops when loss converges
   • Linear classifier, interpretable, fast baseline

2. **Random Forest**:
   • Builds multiple trees independently (n_estimators=100)
   • No iterative refinement needed
   • Parallel tree construction, robust to outliers

3. **Gradient Boosting**:
   • Uses boosting rounds (n_estimators=100)
   • Sequentially builds trees/estimators
   • Each tree corrects errors of previous trees

4. **XGBoost**:
   • Optimized gradient boosting with regularization
   • Uses boosting rounds (n_estimators=100)
   • Handles missing values natively, fast and scalable
""")
        
        self.trained_models = {}
        self.training_info = {}
        
        # Check if hyperparameter tuning is enabled
        tuning_config = self.config.get('hyperparameter_tuning', default={})
        tuning_enabled = tuning_config.get('enabled', False)
        
        if tuning_enabled and self.verbose:
            print(f"\n⚙️  Hyperparameter tuning: ENABLED")
            print(f"   Method: {tuning_config.get('method', 'RandomizedSearchCV')}")
            print(f"   Iterations: {tuning_config.get('n_iter', 50)}")
            print(f"   CV folds: {tuning_config.get('cv', 5)}")
        
        # Map display names back to config names for hyperparameter tuning
        display_to_config = {
            'Logistic Regression': 'LogisticRegression',
            'Random Forest': 'RandomForest',
            'Gradient Boosting': 'GradientBoosting',
            'SVM (RBF)': 'SVM',
            'K-Nearest Neighbors': 'KNN',
            'Naive Bayes': 'NaiveBayes',
            'Neural Network (MLP)': 'NeuralNetwork',
            'XGBoost': 'XGBoost',
            'LightGBM': 'LightGBM'
        }
        
        for idx, (display_name, model) in enumerate(models.items(), 1):
            if self.verbose:
                print(f"\n[{idx}/{len(models)}] Training {display_name}...")
            
            try:
                # Get config name for hyperparameter tuning
                config_name = display_to_config.get(display_name, '')
                
                # Perform hyperparameter tuning if enabled
                if tuning_enabled and config_name:
                    tuned_model = self._tune_hyperparameters(
                        config_name, model, X_train, y_train
                    )
                    model = tuned_model
                else:
                    # Train model directly without tuning
                    model.fit(X_train, y_train)
                
                self.trained_models[display_name] = model
                
                # Store training info
                info = {}
                
                # Check iterations/epochs used
                if hasattr(model, 'n_iter_'):
                    info['n_iter'] = model.n_iter_
                    if 'Neural Network' in display_name:
                        if self.verbose:
                            print(f"   • Epochs used: {model.n_iter_} (max allowed: {model.max_iter})")
                        if hasattr(model, 'best_loss_'):
                            info['best_loss'] = model.best_loss_
                            if self.verbose:
                                print(f"   • Best validation loss: {model.best_loss_:.6f}")
                    elif 'Logistic Regression' in display_name:
                        if self.verbose:
                            print(f"   • Iterations to convergence: {[model.n_iter_]}")
                
                self.training_info[display_name] = info
                
                if self.verbose:
                    print("   ✓ Training completed successfully")
            
            except Exception as e:
                if self.verbose:
                    print(f"   ✗ Training failed: {str(e)}")
                # Continue with other models even if one fails
        
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"✓ Training completed: {len(self.trained_models)}/{len(models)} models trained successfully")
            print("=" * 70)
        
        return self.trained_models
    
    def get_trained_models(self) -> Dict[str, Any]:
        """Get dictionary of trained models."""
        return self.trained_models
    
    def get_training_info(self) -> Dict[str, Dict[str, Any]]:
        """Get training information for all models."""
        return self.training_info


if __name__ == '__main__':
    # Test model trainer
    from src.config import Config
    from src.data_loader import DataLoader
    from src.preprocessor import Preprocessor
    
    config = Config()
    
    # Load and preprocess data
    loader = DataLoader(config)
    X, y, feature_types = loader.load_and_prepare()
    
    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_transformed = preprocessor.fit_transform(
        X_train,
        feature_types['numerical'],
        feature_types['categorical']
    )
    
    # Train models
    trainer = ModelTrainer(config)
    trained_models = trainer.train_all(X_train_transformed, y_train)
    
    print(f"\n✅ Model training completed!")
    print(f"   Trained models: {len(trained_models)}")
    for name in trained_models.keys():
        print(f"   • {name}")

