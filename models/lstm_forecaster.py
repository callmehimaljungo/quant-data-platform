"""
Neural Network forecaster using sklearn MLP
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pickle
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / 'saved_models'
OUTPUT_PATH = GOLD_DIR / 'lstm_predictions_lakehouse'

SEQUENCE_LENGTH = 30
HIDDEN_LAYER_SIZES = (64, 32)
MAX_ITER = 500


class NeuralNetForecaster:
    """MLP-based return forecaster"""
    
    def __init__(self, 
                 sequence_length: int = SEQUENCE_LENGTH,
                 hidden_layers: Tuple = HIDDEN_LAYER_SIZES):
        self.sequence_length = sequence_length
        self.hidden_layers = hidden_layers
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def _create_features_from_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create flattened features from sequences"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            seq = data[i - self.sequence_length:i]
            
            flat = seq.flatten()
            means = seq.mean(axis=0)
            stds = seq.std(axis=0)
            trends = seq[-1] - seq[0]
            
            if len(seq) >= 10:
                recent = seq[-5:].mean(axis=0)
                previous = seq[-10:-5].mean(axis=0)
                momentum = recent - previous
            else:
                momentum = np.zeros(seq.shape[1])
            
            features = np.concatenate([flat, means, stds, trends, momentum])
            
            X.append(features)
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], 
                     target_col: str = 'target_return') -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.preprocessing import StandardScaler
        
        df = df.copy().sort_values('date')
        df = df.dropna(subset=feature_cols + [target_col])
        
        if len(df) < self.sequence_length + 100:
            logger.warning(f"Not enough data: {len(df)} rows")
            return None, None
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        X_features, y_features = self._create_features_from_sequences(X_scaled, y)
        
        return X_features, y_features
    
    def train(self, df: pd.DataFrame, feature_cols: List[str],
              target_col: str = 'target_return',
              test_size: float = 0.2) -> Dict:
        from sklearn.neural_network import MLPRegressor
        
        logger.info("=" * 50)
        logger.info("TRAINING NEURAL NETWORK FORECASTER")
        logger.info("=" * 50)
        
        self.feature_columns = feature_cols
        
        X, y = self.prepare_data(df, feature_cols, target_col)
        
        if X is None:
            return {}
        
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Original features: {len(feature_cols)}")
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train: {len(X_train):,} samples")
        logger.info(f"Test: {len(X_test):,} samples")
        
        logger.info(f"Training MLP with layers {self.hidden_layers}...")
        
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))
        
        direction_correct = np.sum(np.sign(y_pred) == np.sign(y_test))
        direction_accuracy = direction_correct / len(y_test)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'direction_accuracy': float(direction_accuracy),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'iterations': self.model.n_iter_,
        }
        
        logger.info("")
        logger.info("--- TRAINING RESULTS ---")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"Direction Accuracy: {direction_accuracy:.4f}")
        logger.info(f"Iterations: {self.model.n_iter_}")
        
        return metrics
    
    def save(self, path: Optional[Path] = None):
        path = path or MODEL_PATH / 'nn_forecaster.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'hidden_layers': self.hidden_layers,
            }, f)
        
        logger.info(f"[OK] Model saved to {path}")
    
    def load(self, path: Optional[Path] = None):
        path = path or MODEL_PATH / 'nn_forecaster.pkl'
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.sequence_length = data['sequence_length']
        self.hidden_layers = data['hidden_layers']
        
        logger.info(f"[OK] Model loaded from {path}")


def run_lstm_training(test_mode: bool = False) -> Dict:
    """Run Neural Network training pipeline"""
    from models.feature_engineering import engineer_features, get_feature_columns
    from utils import pandas_to_lakehouse
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("NEURAL NETWORK FORECASTER")
    logger.info("=" * 70)
    
    parquet_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if not parquet_path.exists():
        logger.error(f"Data not found: {parquet_path}")
        return {}
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"[OK] Loaded {len(df):,} rows")
    
    top_tickers = df['ticker'].value_counts().head(50 if test_mode else 100).index.tolist()
    df = df[df['ticker'].isin(top_tickers)]
    logger.info(f"Using {len(df):,} rows from {len(top_tickers)} tickers")
    
    logger.info("Creating features...")
    df = engineer_features(df, include_target=True)
    
    feature_cols = [c for c in ['rsi', 'macd', 'bb_pct', 'volatility_20', 
                                 'return_1d', 'return_5d', 'volume_ratio'] 
                    if c in df.columns]
    
    if len(feature_cols) < 3:
        logger.error("Not enough features available")
        return {}
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    best_ticker = df['ticker'].value_counts().index[0]
    df_single = df[df['ticker'] == best_ticker].copy()
    
    logger.info(f"Training on {best_ticker} ({len(df_single):,} rows)")
    
    forecaster = NeuralNetForecaster(
        sequence_length=30,
        hidden_layers=(64, 32)
    )
    
    metrics = forecaster.train(df_single, feature_cols)
    
    if not metrics:
        return {}
    
    forecaster.save()
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[OK] NEURAL NETWORK TRAINING COMPLETED")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Direction Accuracy: {metrics.get('direction_accuracy', 0):.4f}")
    logger.info("=" * 70)
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    run_lstm_training(test_mode=args.test)


if __name__ == "__main__":
    main()
