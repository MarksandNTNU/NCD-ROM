"""
MLflow utilities for model tracking and saving.
"""

import os
import json
import pickle
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import equinox as eqx
import jax

# Try to import mlflow, with fallback
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Using local file-based tracking.")


class ModelTracker:
    """
    Track experiments and save models using MLflow or local files.
    """
    
    def __init__(self, experiment_name: str = "gopro_reconstruction", 
                 tracking_uri: str = None,
                 use_mlflow: bool = True):
        """
        Initialize the tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (e.g., 'file:./mlruns')
            use_mlflow: Whether to use MLflow (falls back to local if not available)
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.run_id = None
        self.save_dir = "saved_models"
        
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        else:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def start_run(self, run_name: str = None):
        """Start a new tracking run."""
        if self.use_mlflow:
            self.run = mlflow.start_run(run_name=run_name)
            self.run_id = self.run.info.run_id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_name:
                self.run_id = f"{run_name}_{self.run_id}"
            self.run_dir = os.path.join(self.save_dir, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)
            self._local_params = {}
            self._local_metrics = {}
        
        return self.run_id
    
    def end_run(self):
        """End the current run."""
        if self.use_mlflow:
            mlflow.end_run()
        else:
            # Save local tracking data
            with open(os.path.join(self.run_dir, 'params.json'), 'w') as f:
                json.dump(self._local_params, f, indent=2)
            with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
                json.dump(self._local_metrics, f, indent=2)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.use_mlflow:
            mlflow.log_params(params)
        else:
            self._local_params.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            for k, v in metrics.items():
                if k not in self._local_metrics:
                    self._local_metrics[k] = []
                # Convert to Python float if JAX/numpy array
                if hasattr(v, 'item'):
                    v = v.item()
                elif hasattr(v, '__float__'):
                    v = float(v)
                self._local_metrics[k].append({'step': step, 'value': v})
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)
    
    def save_model(self, model, model_name: str, additional_data: Dict = None):
        """
        Save an Equinox model.
        
        Args:
            model: Equinox model to save
            model_name: Name for the model
            additional_data: Additional data to save with the model
        """
        if self.use_mlflow:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{model_name}.eqx")
                eqx.tree_serialise_leaves(model_path, model)
                mlflow.log_artifact(model_path)
                
                if additional_data:
                    data_path = os.path.join(tmpdir, f"{model_name}_data.pkl")
                    with open(data_path, 'wb') as f:
                        pickle.dump(additional_data, f)
                    mlflow.log_artifact(data_path)
        else:
            model_path = os.path.join(self.run_dir, f"{model_name}.eqx")
            eqx.tree_serialise_leaves(model_path, model)
            
            if additional_data:
                data_path = os.path.join(self.run_dir, f"{model_name}_data.pkl")
                with open(data_path, 'wb') as f:
                    pickle.dump(additional_data, f)
        
        print(f"Model saved: {model_name}")
        return model_path if not self.use_mlflow else self.run_id
    
    def log_artifact(self, filepath: str):
        """Log a file artifact."""
        if self.use_mlflow:
            mlflow.log_artifact(filepath)
        else:
            import shutil
            shutil.copy(filepath, self.run_dir)
    
    def log_training_history(self, train_losses: list, valid_losses: list, 
                            model_name: str = "model"):
        """Log training history."""
        history = {
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
        
        if self.use_mlflow:
            with tempfile.TemporaryDirectory() as tmpdir:
                history_path = os.path.join(tmpdir, f"{model_name}_history.json")
                with open(history_path, 'w') as f:
                    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
                mlflow.log_artifact(history_path)
        else:
            history_path = os.path.join(self.run_dir, f"{model_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)


def load_model(model_template, model_path: str):
    """
    Load a saved Equinox model.
    
    Args:
        model_template: A model instance with the same structure
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    return eqx.tree_deserialise_leaves(model_path, model_template)


def load_run_data(run_dir: str):
    """
    Load all data from a saved run.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary with params, metrics, and model paths
    """
    data = {}
    
    params_path = os.path.join(run_dir, 'params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data['params'] = json.load(f)
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data['metrics'] = json.load(f)
    
    # Find model files
    model_files = [f for f in os.listdir(run_dir) if f.endswith('.eqx')]
    data['model_paths'] = {f.replace('.eqx', ''): os.path.join(run_dir, f) for f in model_files}
    
    return data


def get_latest_run(save_dir: str = "saved_models"):
    """Get the path to the most recent run."""
    if not os.path.exists(save_dir):
        return None
    
    runs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    if not runs:
        return None
    
    runs.sort(reverse=True)
    return os.path.join(save_dir, runs[0])
