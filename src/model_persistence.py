"""
Model Persistence Module

Handles saving and loading trained models for reuse.
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.config import Config


class ModelPersistence:
    """
    Handles saving and loading trained models.
    
    Provides functionality to persist models to disk and reload them later.
    """
    
    def __init__(self, config: Config):
        """
        Initialize model persistence handler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.results_dir = config.get('output', 'results_dir', default='results')
        self.verbose = config.get('output', 'verbose', default=True)
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            metadata: Optional metadata dictionary (metrics, parameters, etc.)
            
        Returns:
            Path to saved model file
        """
        model_path = os.path.join(self.results_dir, f"{model_name}_model.pkl")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(self.results_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"✓ Model saved: {model_path}")
            if metadata:
                print(f"✓ Metadata saved: {metadata_path}")
        
        return model_path
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = os.path.join(self.results_dir, f"{model_name}_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if self.verbose:
            print(f"✓ Model loaded: {model_path}")
        
        return model
    
    def load_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load model metadata from disk.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = os.path.join(self.results_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def save_all_models(self, trained_models: Dict[str, Any], 
                        evaluation_results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, str]:
        """
        Save all trained models and their metadata.
        
        Args:
            trained_models: Dictionary of trained models
            evaluation_results: Optional evaluation results dictionary
            
        Returns:
            Dictionary mapping model names to saved file paths
        """
        saved_paths = {}
        
        for model_name, model in trained_models.items():
            metadata = None
            if evaluation_results and model_name in evaluation_results:
                metadata = evaluation_results[model_name]
            
            path = self.save_model(model, model_name, metadata)
            saved_paths[model_name] = path
        
        if self.verbose:
            print(f"\n✓ Saved {len(saved_paths)} models to {self.results_dir}")
        
        return saved_paths

