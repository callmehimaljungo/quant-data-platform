"""
XGBoost classifier for price direction prediction
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pickle
from datetime import datetime
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from config import SILVER_DIR, GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / 'saved_models'
OUTPUT_PATH = GOLD_DIR / 'ml_predictions_lakehouse'


class XGBoostDirectionClassifier:
    """Predict next-day price direction using XGBoost"""
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data and extract features/target"""
        df = df.copy()
        df = df.dropna(subset=feature_cols + ['target_direction'])
        X = df[feature_cols]
        y = df['target_direction']
        return X, y
    
    def train(self, df: pd.DataFrame, feature_cols: list, 
              train_end_date: Optional[str] = None,
              test_size: float = 0.2) -> Dict:
        """Train model with time-based split"""
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        except ImportError:
            logger.error("XGBoost not installed. Run: pip install xgboost scikit-learn")
            return {}
        
        logger.info("=" * 50)
        logger.info("TRAINING XGBOOST CLASSIFIER")
        logger.info("=" * 50)
        
        self.feature_columns = feature_cols
        X, y = self.prepare_data(df, feature_cols)
        logger.info(f"Data: {len(X):,} samples, {len(feature_cols)} features")
        
        if len(X) < 1000:
            logger.warning("Not enough data for reliable training!")
        
        # Time-based split to avoid look-ahead bias
        if train_end_date:
            train_mask = df['date'] <= train_end_date
            X_train = X[train_mask[:len(X)]]
            y_train = y[train_mask[:len(y)]]
            X_test = X[~train_mask[:len(X)]]
            y_test = y[~train_mask[:len(y)]]
        else:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train):,} samples")
        logger.info(f"Test: {len(X_test):,} samples")
        
        logger.info("Training XGBoost...")
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("")
        logger.info("--- TRAINING RESULTS ---")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info("")
        logger.info("Top 5 Features:")
        for idx, row in self.feature_importance.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df = df.copy()
        X = df[self.feature_columns].copy()
        X = X.fillna(X.mean())
        
        df['xgb_prediction'] = self.model.predict(X)
        df['xgb_probability'] = self.model.predict_proba(X)[:, 1]
        
        df['xgb_signal'] = df['xgb_probability'].apply(
            lambda p: 'STRONG_BUY' if p > 0.7 else 
                      'BUY' if p > 0.55 else
                      'HOLD' if p > 0.45 else
                      'SELL' if p > 0.3 else 'STRONG_SELL'
        )
        
        return df
    
    def save(self, path: Optional[Path] = None):
        path = path or MODEL_PATH / 'xgboost_classifier.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'params': self.params,
            }, f)
        
        logger.info(f"[OK] Model saved to {path}")
    
    def load(self, path: Optional[Path] = None):
        path = path or MODEL_PATH / 'xgboost_classifier.pkl'
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.feature_importance = data['feature_importance']
        self.params = data['params']
        
        logger.info(f"[OK] Model loaded from {path}")


def run_xgboost_training(test_mode: bool = False) -> Dict:
    """Run XGBoost training pipeline"""
    from models.feature_engineering import engineer_features, get_feature_columns
    from utils import pandas_to_lakehouse
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("XGBOOST DIRECTION CLASSIFIER")
    logger.info("=" * 70)
    
    parquet_path = SILVER_DIR / 'enriched_stocks.parquet'
    
    if not parquet_path.exists():
        logger.error(f"Data not found: {parquet_path}")
        return {}
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"[OK] Loaded {len(df):,} rows")
    
    if test_mode:
        top_tickers = df['ticker'].value_counts().head(20).index.tolist()
        df = df[df['ticker'].isin(top_tickers)]
        logger.info(f"[TEST MODE] Using {len(df):,} rows from {len(top_tickers)} tickers")
    
    logger.info("Creating features...")
    df = engineer_features(df, include_target=True)
    
    feature_cols = get_feature_columns()
    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_features)} features")
    
    classifier = XGBoostDirectionClassifier()
    metrics = classifier.train(df, available_features)
    
    if not metrics:
        return {}
    
    classifier.save()
    
    logger.info("Making predictions on latest data...")
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()
    
    if len(df_latest) > 0:
        df_predictions = classifier.predict(df_latest)
        
        output = df_predictions[['ticker', 'date', 'close', 'xgb_prediction', 
                                  'xgb_probability', 'xgb_signal']].copy()
        output['model'] = 'xgboost'
        output['created_at'] = datetime.now()
        
        pandas_to_lakehouse(output, OUTPUT_PATH, mode='overwrite')
        
        logger.info(f"[OK] Predictions saved for {len(output):,} tickers")
        
        print("\n--- TOP BUY SIGNALS ---")
        top_buys = output.nlargest(10, 'xgb_probability')
        print(top_buys[['ticker', 'xgb_probability', 'xgb_signal']].to_string(index=False))
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[OK] XGBOOST TRAINING COMPLETED")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info("=" * 70)
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    run_xgboost_training(test_mode=args.test)


if __name__ == "__main__":
    main()
