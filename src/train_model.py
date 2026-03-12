"""
Train the ML ensemble model on historical data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.data_collector import SyntheticDataGenerator, prepare_training_data
from src.features import FeatureEngineer
from src.ml_ensemble import LSTMEnsembleModel, create_sequences

def train_ensemble_model():
    """
    Main training pipeline:
    1. Generate synthetic training data (30 markets)
    2. Add 38 technical indicators
    3. Create sequences for LSTM
    4. Train ensemble model
    5. Save model and evaluate
    """
    
    print("=" * 60)
    print("TRAINING ML ENSEMBLE MODEL")
    print("=" * 60)
    
    # ===== STEP 1: Generate synthetic data =====
    print("\n[1/5] Generating synthetic market data...")
    markets_data = SyntheticDataGenerator.generate_multiple_markets(
        n_markets=30,
        days=365  # 1 year of hourly data
    )
    print(f"✓ Generated {len(markets_data)} markets with 1 year of data each")
    
    # ===== STEP 2: Add technical indicators =====
    print("\n[2/5] Engineering features (38 indicators)...")
    feature_engineer = FeatureEngineer()
    
    all_data = []
    
    for market_id, df in markets_data.items():
        df_features = feature_engineer.add_features(df)
        all_data.append(df_features)
    
    # Combine all market data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✓ Added 38 indicators to {len(combined_df)} total records")
    
    # ===== STEP 3: Split train/val/test =====
    print("\n[3/5] Splitting data (train 64% / val 16% / test 20%)...")
    train_df, val_df, test_df = prepare_training_data(combined_df, test_split=0.2)
    
    print(f"  Train: {len(train_df)} records")
    print(f"  Val:   {len(val_df)} records")
    print(f"  Test:  {len(test_df)} records")
    
    # ===== STEP 4: Prepare sequences =====
    print("\n[4/5] Creating sequences for LSTM (window_size=30)...")
    
    feature_names = feature_engineer.get_feature_names()
    
    # Normalize using training data only
    train_scaled, val_scaled, scaler = feature_engineer.normalize_features(
        train_df[feature_names],
        val_df[feature_names]
    )
    
    # Create sequences
    X_train, y_train = create_sequences(
        np.hstack([train_scaled, train_df[['Target_1D']].values]),
        window_size=30
    )
    X_val, y_val = create_sequences(
        np.hstack([val_scaled, val_df[['Target_1D']].values]),
        window_size=30
    )
    
    print(f"  Train sequences: {X_train.shape}")
    print(f"  Val sequences:   {X_val.shape}")
    
    # ===== STEP 5: Train model =====
    print("\n[5/5] Training LSTM ensemble (50 epochs)...")
    
    model = LSTMEnsembleModel(
        window_size=30,
        n_features=len(feature_names)
    )
    
    model.build_model()
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # ===== EVALUATION =====
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluate on test set
    test_scaled = scaler.transform(test_df[feature_names])
    X_test, y_test = create_sequences(
        np.hstack([test_scaled, test_df[['Target_1D']].values]),
        window_size=30
    )
    
    if len(X_test) > 0:
        test_loss, test_acc, test_auc = model.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Results (unseen data from 2024-2025):")
        print(f"  Loss:     {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  AUC:      {test_auc:.4f}")
    
    # MC dropout predictions on test set
    print(f"\nMonte Carlo Dropout Evaluation (50 forward passes):")
    mean, std, _ = model.mc_predict(X_test[:100])  # First 100 test samples
    
    print(f"  Mean confidence:     {mean.mean():.4f}")
    print(f"  Std (uncertainty):   {std.mean():.4f}")
    print(f"  High confidence (>0.7): {(mean > 0.7).sum()} / {len(mean)}")
    
    # ===== SAVE MODEL =====
    print("\n" + "=" * 60)
    
    Path('models').mkdir(exist_ok=True)
    model.save('models')
    
    print("✓ Model saved to models/ensemble_model.h5")
    print("✓ Config saved to models/model_config.json")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return model, feature_engineer

if __name__ == "__main__":
    train_ensemble_model()
