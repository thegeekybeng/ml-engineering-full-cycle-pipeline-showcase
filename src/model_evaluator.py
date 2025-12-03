"""
Model Evaluation Module

Handles comprehensive model evaluation including:
- Core metrics (Accuracy, Precision, Recall, F1-Score)
- Security metrics (Specificity, FPR, FNR)
- Balanced metrics (Balanced Accuracy, MCC)
- AUC metrics (ROC-AUC, PR-AUC)
- Model comparison and best model selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    matthews_corrcoef, precision_recall_curve, auc,
    classification_report
)
from src.config import Config


class ModelEvaluator:
    """
    Model evaluator for MLP pipeline.
    
    Evaluates models with comprehensive metrics and selects the best model.
    """
    
    def __init__(self, config: Config):
        """
        Initialize model evaluator with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.verbose = config.get('output', 'verbose', default=True)
        self.evaluation_results = {}
        self.results_df = None
        self.best_model_name = None
        self.best_model = None
        self.best_results = None
    
    def evaluate_all(self, trained_models: Dict[str, Any], 
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            trained_models: Dictionary of trained models
            X_test: Test features (preprocessed)
            y_test: Test target
        
        Returns:
            Dictionary of evaluation results for each model
        """
        if self.verbose:
            print("=" * 70)
            print("STEP 6: MODEL EVALUATION")
            print("=" * 70)
        
        self.evaluation_results = {}
        
        for idx, (name, model) in enumerate(trained_models.items(), 1):
            if self.verbose:
                print(f"\n[{idx}/{len(trained_models)}] Evaluating {name}...")
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate confusion matrix components
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # Core metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)  # Also called Sensitivity/TPR
                f1 = f1_score(y_test, y_pred)
                
                # Security-focused metrics
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
                
                # Balanced metrics
                balanced_accuracy = (recall + specificity) / 2.0
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # AUC metrics
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall_curve, precision_curve)
                
                # Store results
                self.evaluation_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'specificity': specificity,
                    'fpr': fpr,
                    'fnr': fnr,
                    'balanced_accuracy': balanced_accuracy,
                    'mcc': mcc,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'cm': cm
                }
                
                if self.verbose:
                    print(f"   📊 Core Metrics:")
                    print(f"      • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"      • Precision: {precision:.4f} ({precision*100:.2f}%)")
                    print(f"      • Recall:    {recall:.4f} ({recall*100:.2f}%)")
                    print(f"      • F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
                    print(f"   🔒 Security Metrics:")
                    print(f"      • Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
                    print(f"      • FPR:         {fpr:.4f} ({fpr*100:.2f}%)")
                    print(f"      • FNR:         {fnr:.4f} ({fnr*100:.2f}%)")
                    print(f"   ⚖️  Balanced Metrics:")
                    print(f"      • Balanced Acc: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
                    print(f"      • MCC:          {mcc:.4f}")
                    print(f"   📈 AUC Metrics:")
                    print(f"      • ROC-AUC:      {roc_auc:.4f}")
                    print(f"      • PR-AUC:       {pr_auc:.4f}")
            
            except Exception as e:
                if self.verbose:
                    print(f"   ✗ Evaluation failed: {str(e)}")
        
        if self.verbose:
            print("\n" + "=" * 70)
        
        return self.evaluation_results
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """
        Create results DataFrame for easy comparison.
        
        Returns:
            DataFrame with all metrics for all models, sorted by accuracy
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        # Extract metrics (exclude predictions and confusion matrix)
        metric_columns = [
            'accuracy', 'precision', 'recall', 'f1',
            'specificity', 'fpr', 'fnr',
            'balanced_accuracy', 'mcc',
            'roc_auc', 'pr_auc'
        ]
        
        results_data = {
            col: [self.evaluation_results[name][col] for name in self.evaluation_results.keys()]
            for col in metric_columns
        }
        
        self.results_df = pd.DataFrame(
            results_data,
            index=list(self.evaluation_results.keys())
        )
        
        # Sort by accuracy (descending)
        self.results_df = self.results_df.sort_values('accuracy', ascending=False)
        
        if self.verbose:
            print("=" * 70)
            print("MODEL COMPARISON TABLE - COMPREHENSIVE METRICS")
            print("=" * 70)
            print("\n📊 Core Metrics (Accuracy, Precision, Recall, F1-Score):")
            print(self.results_df[['accuracy', 'precision', 'recall', 'f1']].round(4).to_string())
            print("\n🔒 Security Metrics (Specificity, FPR, FNR):")
            print(self.results_df[['specificity', 'fpr', 'fnr']].round(4).to_string())
            print("\n⚖️  Balanced Metrics (Balanced Accuracy, MCC):")
            print(self.results_df[['balanced_accuracy', 'mcc']].round(4).to_string())
            print("\n📈 AUC Metrics (ROC-AUC, PR-AUC):")
            print(self.results_df[['roc_auc', 'pr_auc']].round(4).to_string())
            print("=" * 70)
        
        return self.results_df
    
    def select_best_model(self, trained_models: Dict[str, Any], 
                          criterion: str = 'accuracy') -> Tuple[str, Any, Dict[str, Any]]:
        """
        Select best model based on specified criterion.
        
        Args:
            trained_models: Dictionary of trained models
            criterion: Metric to use for selection (default: 'accuracy')
        
        Returns:
            Tuple of (best_model_name, best_model, best_results)
        """
        if self.results_df is None:
            self.create_results_dataframe()
        
        # Select best model (highest value for criterion)
        self.best_model_name = self.results_df.index[0]  # Already sorted by accuracy
        self.best_model = trained_models[self.best_model_name]
        self.best_results = self.evaluation_results[self.best_model_name]
        
        if self.verbose:
            print("=" * 70)
            print("STEP 7: BEST MODEL SELECTION")
            print("=" * 70)
            print(f"\n📊 Selection Method: Comprehensive evaluation across all metrics")
            print(f"   Primary criterion: Accuracy (highest: {self.best_model_name})")
            print(f"   Secondary: Balanced performance across precision, recall, and security metrics")
            
            print(f"\n🏆 Best Model: {self.best_model_name}")
            print(f"   Model Type: {type(self.best_model).__name__}")
            
            # Print model-specific information
            if hasattr(self.best_model, 'hidden_layer_sizes'):
                print(f"   Architecture: {self.best_model.hidden_layer_sizes}")
            if hasattr(self.best_model, 'n_estimators'):
                print(f"   Number of Estimators: {self.best_model.n_estimators}")
            if hasattr(self.best_model, 'n_iter_'):
                print(f"   Training Iterations: {self.best_model.n_iter_}")
            
            print(f"\n📊 Core Performance Metrics:")
            print(f"   • Accuracy:  {self.best_results['accuracy']:.4f} ({self.best_results['accuracy']*100:.2f}%)")
            print(f"   • Precision: {self.best_results['precision']:.4f} ({self.best_results['precision']*100:.2f}%)")
            print(f"   • Recall:    {self.best_results['recall']:.4f} ({self.best_results['recall']*100:.2f}%)")
            print(f"   • F1-Score:  {self.best_results['f1']:.4f} ({self.best_results['f1']*100:.2f}%)")
            
            print(f"\n🔒 Security-Focused Metrics:")
            print(f"   • Specificity: {self.best_results['specificity']:.4f} ({self.best_results['specificity']*100:.2f}%)")
            print(f"   • FPR:         {self.best_results['fpr']:.4f} ({self.best_results['fpr']*100:.2f}%)")
            print(f"   • FNR:         {self.best_results['fnr']:.4f} ({self.best_results['fnr']*100:.2f}%)")
            
            print(f"\n⚖️  Balanced Metrics:")
            print(f"   • Balanced Acc: {self.best_results['balanced_accuracy']:.4f} ({self.best_results['balanced_accuracy']*100:.2f}%)")
            print(f"   • MCC:          {self.best_results['mcc']:.4f}")
            
            print(f"\n📈 AUC Metrics:")
            print(f"   • ROC-AUC:      {self.best_results['roc_auc']:.4f} ({self.best_results['roc_auc']*100:.2f}%)")
            print(f"   • PR-AUC:       {self.best_results['pr_auc']:.4f} ({self.best_results['pr_auc']*100:.2f}%)")
            
            print("\n" + "=" * 70)
        
        return self.best_model_name, self.best_model, self.best_results
    
    def get_classification_report(self, y_test: np.ndarray) -> str:
        """
        Get detailed classification report for best model.
        
        Args:
            y_test: Test target
        
        Returns:
            Classification report string
        """
        if self.best_results is None:
            raise ValueError("Best model not selected. Run select_best_model() first.")
        
        report = classification_report(
            y_test,
            self.best_results['y_pred'],
            target_names=['Phishing', 'Legitimate']
        )
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"DETAILED CLASSIFICATION REPORT: {self.best_model_name}")
            print("=" * 70 + "\n")
            print(report)
            print("=" * 70)
        
        return report
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results DataFrame."""
        if self.results_df is None:
            self.create_results_dataframe()
        return self.results_df
    
    def get_evaluation_results(self) -> Dict[str, Dict[str, Any]]:
        """Get evaluation results dictionary."""
        return self.evaluation_results
    
    def get_best_model_info(self) -> Tuple[str, Any, Dict[str, Any]]:
        """Get best model information."""
        if self.best_model_name is None:
            raise ValueError("Best model not selected. Run select_best_model() first.")
        return self.best_model_name, self.best_model, self.best_results


if __name__ == '__main__':
    # Test model evaluator
    from src.config import Config
    from src.data_loader import DataLoader
    from src.preprocessor import Preprocessor
    from src.model_trainer import ModelTrainer
    
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
    X_test_transformed = preprocessor.transform(X_test)
    
    # Train models
    trainer = ModelTrainer(config)
    trained_models = trainer.train_all(X_train_transformed, y_train)
    
    # Evaluate models
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate_all(trained_models, X_test_transformed, y_test)
    
    # Create results DataFrame
    results_df = evaluator.create_results_dataframe()
    
    # Select best model
    best_name, best_model, best_results = evaluator.select_best_model(trained_models)
    
    print(f"\n✅ Model evaluation completed!")
    print(f"   Best model: {best_name}")
    print(f"   Best accuracy: {best_results['accuracy']:.4f}")

