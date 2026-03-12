"""
ML Ensemble Bot - LSTM with Monte Carlo Dropout for uncertainty quantification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime
from typing import Tuple

class MCDropout(keras.layers.Dropout):
    """Dropout layer that stays active during inference for uncertainty estimation."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

class LSTMEnsembleModel:
    """
    LSTM ensemble with Monte Carlo Dropout.
    - Conv1D for local patterns
    - LSTM for long-term dependencies  
    - MC Dropout for uncertainty quantification (50 stochastic forward passes)
    - Sigmoid output = probability (0-1)
    """
    
    def __init__(self, window_size: int = 30, n_features: int = 34):
        self.window_size = window_size
        self.n_features = n_features
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """Build LSTM ensemble architecture."""
        
        model = Sequential([
            layers.Input(shape=(self.window_size, self.n_features)),
            
            # Conv1D: Extract local patterns (trend changes, volatility spikes)
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            MCDropout(0.2),
            
            # LSTM: Long-term dependencies
            layers.LSTM(64, return_sequences=False, activation='tanh'),
            layers.BatchNormalization(),
            MCDropout(0.2),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            MCDropout(0.2),
            layers.Dense(16, activation='relu'),
            
            # Output: probability (0-1)
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Train model on data.
        
        Args:
            X_train: Training features (samples, window_size, n_features)
            y_train: Training targets (samples,)
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Training history dict
        """
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                min_delta=1e-7,
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        return history.history
    
    def mc_predict(self, X: np.ndarray, n_samples: int = 50) -> Tuple:
        """
        Monte Carlo dropout prediction.
        Run inference 50 times with different dropout patterns.
        
        Returns:
            (mean_confidence, std_uncertainty, all_predictions)
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Run model n_samples times with dropout active
        predictions = np.array([
            self.model(X, training=True).numpy()
            for _ in range(n_samples)
        ])
        
        # predictions shape: (n_samples, batch_size, 1)
        # Remove the last singleton dimension: (n_samples, batch_size)
        predictions = predictions[:, :, 0]
        
        # Calculate mean and std across samples (axis 0)
        mean = predictions.mean(axis=0)  # shape: (batch_size,)
        std = predictions.std(axis=0)    # shape: (batch_size,)
        
        return mean, std, predictions
    
    def predict_with_confidence(self, X: np.ndarray, confidence_threshold: float = 0.7) -> dict:
        """
        Predict with confidence filtering.
        Only output signals when mean > threshold AND std is low.
        """
        
        mean, std, _ = self.mc_predict(X)
        
        # Signals: BUY if high confidence, HOLD otherwise
        signals = []
        for m, s in zip(mean, std):
            if m > confidence_threshold and s < 0.15:  # Low uncertainty
                signals.append('BUY')
            else:
                signals.append('HOLD')
        
        return {
            'signals': signals,
            'confidence': mean,
            'uncertainty': std
        }
    
    def save(self, path: str):
        """Save model weights and config."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(f"{path}/ensemble_model.h5")
        with open(f"{path}/model_config.json", 'w') as f:
            json.dump({
                'window_size': self.window_size,
                'n_features': self.n_features,
                'trained_at': datetime.now().isoformat()
            }, f)
    
    def load(self, path: str):
        """Load pre-trained model."""
        self.model = keras.models.load_model(
            f"{path}/ensemble_model.h5",
            custom_objects={'MCDropout': MCDropout}
        )
        with open(f"{path}/model_config.json", 'r') as f:
            config = json.load(f)
            self.window_size = config['window_size']
            self.n_features = config['n_features']


def create_sequences(data: np.ndarray, window_size: int = 30) -> Tuple:
    """
    Create sliding window sequences for LSTM.
    
    Input: (n_samples, n_features) - flattened time series
    Output: (n_sequences, window_size, n_features)
    """
    
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, -1])  # Target is last column
    
    return np.array(X), np.array(y)
