"""
Main Pipeline Entry Point.

This module provides the primary executable entry point for the phishing
classification machine learning pipeline. It orchestrates the full lifecycle
from data loading and preprocessing through model training, evaluation, and
optional persistence, using a configuration-driven design.

Usage:
    python3 src/pipeline.py --config config.yaml
    python3 src/pipeline.py --model RandomForest --test-size 0.2
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, create_parser


class MLPipeline:
    """
    Main Machine Learning Pipeline orchestrator.
    
    Coordinates data loading, preprocessing, model training, and evaluation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.verbose = config.get('output', 'verbose', default=True)
        
        if self.verbose:
            print("=" * 70)
            print("MACHINE LEARNING PIPELINE INITIALIZED")
            print("=" * 70)
            print(f"Config loaded: {len(self.config.to_dict())} sections")
            print(f"Database URL: {self.config.get('data', 'db_url')}")
            print(f"Test Size: {self.config.get('data', 'test_size')}")
            print(f"Random State: {self.config.get('data', 'random_state')}")
            print(f"Enabled Models: {len(self.config.get('models', 'enabled'))} models")
            print("=" * 70)
    
    def run(self):
        """
        Execute the complete ML pipeline.
        
        Pipeline steps:
        1. Load data from SQLite database
        2. Preprocess data (scaling, encoding, etc.)
        3. Split into train/test sets
        4. Train models
        5. Evaluate models
        6. Compare and select best model
        7. Perform final validation and reporting
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("PIPELINE EXECUTION STARTED")
            print("=" * 70)
        
        try:
            # Step 1: Load Data
            if self.verbose:
                print("\n[Step 1/7] Loading data from SQLite database...")
            from src.data_loader import DataLoader
            data_loader = DataLoader(self.config)
            X, y, feature_types = data_loader.load_and_prepare()
            
            if self.verbose:
                print(f"   ✓ Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
                print(f"   ✓ Numerical features: {len(feature_types['numerical'])}")
                print(f"   ✓ Categorical features: {len(feature_types['categorical'])}")
            
            # Step 2: Preprocess Data and Split
            if self.verbose:
                print("\n[Step 2/7] Preprocessing data and splitting into train/test sets...")
            from src.preprocessor import Preprocessor
            preprocessor = Preprocessor(self.config)
            
            # Split data
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
            
            # Fit and transform training data
            X_train_transformed = preprocessor.fit_transform(
                X_train,
                feature_types['numerical'],
                feature_types['categorical']
            )
            
            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)
            
            if self.verbose:
                print(f"   ✓ Preprocessing complete")
                print(f"   ✓ Training shape: {X_train_transformed.shape}")
                print(f"   ✓ Test shape: {X_test_transformed.shape}")
            
            # Step 4: Train Models
            if self.verbose:
                print("\n[Step 4/7] Training models...")
            from src.model_trainer import ModelTrainer
            trainer = ModelTrainer(self.config)
            trained_models = trainer.train_all(X_train_transformed, y_train)
            
            if self.verbose:
                print(f"   ✓ Models trained: {len(trained_models)}")
            
            # Step 5: Evaluate Models
            if self.verbose:
                print("\n[Step 5/7] Evaluating models...")
            from src.model_evaluator import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            evaluation_results = evaluator.evaluate_all(trained_models, X_test_transformed, y_test)
            
            # Step 6: Compare Models and Select Best
            if self.verbose:
                print("\n[Step 6/7] Comparing models and selecting best...")
            results_df = evaluator.create_results_dataframe()
            best_model_name, best_model, best_results = evaluator.select_best_model(trained_models)
            
            # Step 7: Final validation and reporting
            if self.verbose:
                print("\n[Step 7/7] Generating final validation and reporting...")
            classification_report = evaluator.get_classification_report(y_test)
            
            # Step 8: Save Models (Optional - if enabled in config)
            if self.config.get('output', 'save_models', default=False):
                if self.verbose:
                    print("\n[Step 8/8] Saving trained models...")
                from src.model_persistence import ModelPersistence
                persistence = ModelPersistence(self.config)
                persistence.save_all_models(trained_models, evaluation_results)
            
            # Store for potential use
            self.evaluator = evaluator
            self.best_model_name = best_model_name
            self.best_model = best_model
            self.best_results = best_results
            
            if self.verbose:
                print("\n" + "=" * 70)
                print("PIPELINE EXECUTION COMPLETED")
                print("=" * 70)
                print(f"\n✅ Machine Learning Pipeline completed successfully!")
                print(f"   • Data loaded and preprocessed")
                print(f"   • {len(trained_models)} models trained")
                print(f"   • All models evaluated with comprehensive metrics")
                print(f"   • Best model selected: {self.best_model_name}")
                print(f"   • Best accuracy: {self.best_results['accuracy']:.2%}")
                print("=" * 70)
        
        except KeyboardInterrupt:
            if self.verbose:
                print("\n\n⚠️  Pipeline execution interrupted by user")
            raise
        except Exception as e:
            if self.verbose:
                print(f"\n❌ ERROR: Pipeline execution failed: {e}")
                import traceback
                print("\nFull traceback:")
                traceback.print_exc()
            raise


def main():
    """Main entry point for command-line execution."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(config_path=args.config, args=args)
    except Exception as e:
        print(f"❌ ERROR: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize and run pipeline
    try:
        pipeline = MLPipeline(config)
        pipeline.run()
    except Exception as e:
        print(f"❌ ERROR: Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

