"""
Data Quality Monitoring Utility

Purpose:
- Track row counts and data statistics across layers
- Log data quality metrics for lineage tracking
- Compare data between layers to identify data loss
- Generate quality reports

Usage:
    from utils.data_quality import DataQualityMonitor
    
    monitor = DataQualityMonitor()
    monitor.log_layer('bronze', df, 'after_ingestion')
    monitor.log_layer('silver', df_clean, 'after_cleaning')
    monitor.compare_layers('bronze', 'silver')
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Monitor and track data quality across pipeline layers.
    
    Features:
    - Log statistics at each processing step
    - Track row counts, null rates, date ranges
    - Compare layers to identify data loss
    - Export metrics to JSON for audit trail
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize DataQualityMonitor.
        
        Args:
            log_dir: Directory to save quality logs. Defaults to ./data/quality_logs/
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / 'data' / 'quality_logs'
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[Dict[str, Any]] = []
        self.log_file = self.log_dir / 'quality_log.jsonl'
    
    def log_layer(self, layer: str, df: pd.DataFrame, step: str, 
                  additional_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Log quality metrics for a DataFrame at a specific layer/step.
        
        Args:
            layer: Layer name ('bronze', 'silver', 'gold')
            df: DataFrame to analyze
            step: Processing step description
            additional_info: Optional additional metadata
            
        Returns:
            dict: Quality metrics
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'layer': layer,
            'step': step,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Null analysis
        null_counts = df.isnull().sum()
        metric['null_counts'] = null_counts[null_counts > 0].to_dict()
        metric['total_nulls'] = int(null_counts.sum())
        metric['null_rate'] = float(null_counts.sum() / (len(df) * len(df.columns))) if len(df) > 0 else 0
        
        # Date range (if date column exists)
        date_cols = ['date', 'Date', 'timestamp', 'datetime']
        for col in date_cols:
            if col in df.columns:
                try:
                    metric['date_min'] = str(df[col].min())
                    metric['date_max'] = str(df[col].max())
                    break
                except:
                    pass
        
        # Ticker count (if ticker column exists)
        ticker_cols = ['ticker', 'Ticker', 'symbol', 'Symbol']
        for col in ticker_cols:
            if col in df.columns:
                metric['ticker_count'] = int(df[col].nunique())
                break
        
        # Duplicate check
        metric['duplicate_rows'] = int(df.duplicated().sum())
        
        # Additional info
        if additional_info:
            metric['additional'] = additional_info
        
        # Save to memory and file
        self.metrics.append(metric)
        self._write_to_log(metric)
        
        # Log summary
        logger.info(f"📊 [{layer.upper()}] {step}")
        logger.info(f"   Rows: {metric['row_count']:,} | Columns: {metric['column_count']}")
        if metric.get('ticker_count'):
            logger.info(f"   Tickers: {metric['ticker_count']:,}")
        if metric.get('date_min'):
            logger.info(f"   Date range: {metric['date_min']} to {metric['date_max']}")
        if metric['total_nulls'] > 0:
            logger.warning(f"   ⚠️ Nulls: {metric['total_nulls']:,} ({metric['null_rate']:.2%})")
        if metric['duplicate_rows'] > 0:
            logger.warning(f"   ⚠️ Duplicates: {metric['duplicate_rows']:,}")
        
        return metric
    
    def compare_layers(self, layer1: str, layer2: str, 
                       step1: Optional[str] = None, 
                       step2: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare metrics between two layers to identify data changes.
        
        Args:
            layer1: First layer name (e.g., 'bronze')
            layer2: Second layer name (e.g., 'silver')
            step1: Optional specific step from layer1
            step2: Optional specific step from layer2
            
        Returns:
            dict: Comparison results
        """
        # Get latest metrics for each layer
        m1_list = [m for m in self.metrics if m['layer'] == layer1]
        m2_list = [m for m in self.metrics if m['layer'] == layer2]
        
        if step1:
            m1_list = [m for m in m1_list if m['step'] == step1]
        if step2:
            m2_list = [m for m in m2_list if m['step'] == step2]
        
        if not m1_list or not m2_list:
            logger.warning(f"Cannot compare: missing metrics for {layer1} or {layer2}")
            return {}
        
        m1 = m1_list[-1]  # Latest metric for layer1
        m2 = m2_list[-1]  # Latest metric for layer2
        
        rows_diff = m1['row_count'] - m2['row_count']
        drop_rate = rows_diff / m1['row_count'] if m1['row_count'] > 0 else 0
        
        comparison = {
            'layer1': layer1,
            'layer2': layer2,
            'rows_layer1': m1['row_count'],
            'rows_layer2': m2['row_count'],
            'rows_dropped': rows_diff,
            'drop_rate': drop_rate,
            'columns_added': list(set(m2.get('columns', [])) - set(m1.get('columns', []))),
            'columns_removed': list(set(m1.get('columns', [])) - set(m2.get('columns', []))),
        }
        
        # Log comparison
        logger.info(f"\n📊 LAYER COMPARISON: {layer1.upper()} → {layer2.upper()}")
        logger.info(f"   {layer1}: {m1['row_count']:,} rows")
        logger.info(f"   {layer2}: {m2['row_count']:,} rows")
        logger.info(f"   Rows dropped: {rows_diff:,} ({drop_rate:.2%})")
        
        if drop_rate > 0.05:
            logger.warning(f"   ⚠️ HIGH DROP RATE: {drop_rate:.2%} > 5% threshold")
        elif drop_rate > 0.01:
            logger.info(f"   ℹ️ Moderate drop rate: {drop_rate:.2%}")
        else:
            logger.info(f"   ✅ Low drop rate: {drop_rate:.2%}")
        
        if comparison['columns_added']:
            logger.info(f"   Columns added: {comparison['columns_added']}")
        if comparison['columns_removed']:
            logger.info(f"   Columns removed: {comparison['columns_removed']}")
        
        return comparison
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all logged metrics as DataFrame"""
        if not self.metrics:
            return pd.DataFrame()
        
        summary = []
        for m in self.metrics:
            summary.append({
                'timestamp': m['timestamp'],
                'layer': m['layer'],
                'step': m['step'],
                'row_count': m['row_count'],
                'column_count': m['column_count'],
                'ticker_count': m.get('ticker_count'),
                'null_rate': m.get('null_rate', 0),
                'duplicate_rows': m.get('duplicate_rows', 0),
            })
        
        return pd.DataFrame(summary)
    
    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Save full quality report to JSON file.
        
        Args:
            output_path: Output file path. Defaults to quality_report_{timestamp}.json
            
        Returns:
            Path: Path to saved report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.log_dir / f'quality_report_{timestamp}.json'
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_checkpoints': len(self.metrics),
            'metrics': self.metrics,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Quality report saved to: {output_path}")
        return output_path
    
    def _write_to_log(self, metric: Dict[str, Any]) -> None:
        """Append metric to JSONL log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metric, default=str) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write to log file: {e}")
    
    def load_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Load historical metrics from log file.
        
        Args:
            last_n: Load only last N entries. None for all.
            
        Returns:
            List of metric dictionaries
        """
        if not self.log_file.exists():
            return []
        
        metrics = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except:
                    continue
        
        if last_n:
            metrics = metrics[-last_n:]
        
        return metrics


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================
def quick_quality_check(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
    """
    Quick one-off quality check without monitor instance.
    
    Args:
        df: DataFrame to check
        name: Name for logging
        
    Returns:
        dict: Quality metrics
    """
    print(f"\n{'='*60}")
    print(f"QUALITY CHECK: {name}")
    print(f"{'='*60}")
    
    print(f"\n📊 Basic Stats:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Nulls
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    null_cols = null_counts[null_counts > 0]
    
    print(f"\n🔍 Null Analysis:")
    if len(null_cols) == 0:
        print("   ✅ No null values found")
    else:
        print(f"   ⚠️ Total nulls: {total_nulls:,}")
        for col, count in null_cols.items():
            pct = count / len(df) * 100
            print(f"      {col}: {count:,} ({pct:.1f}%)")
    
    # Duplicates
    dups = df.duplicated().sum()
    print(f"\n🔄 Duplicates: {dups:,}")
    
    # Date range
    for col in ['date', 'Date']:
        if col in df.columns:
            print(f"\n📅 Date Range:")
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")
            break
    
    # Ticker stats
    for col in ['ticker', 'Ticker']:
        if col in df.columns:
            print(f"\n📈 Ticker Stats:")
            print(f"   Unique tickers: {df[col].nunique():,}")
            ticker_counts = df[col].value_counts()
            print(f"   Min records per ticker: {ticker_counts.min():,}")
            print(f"   Max records per ticker: {ticker_counts.max():,}")
            print(f"   Median records per ticker: {ticker_counts.median():.0f}")
            break
    
    print(f"\n{'='*60}\n")
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'total_nulls': int(total_nulls),
        'duplicates': int(dups),
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================
if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=1000),
        'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], 1000),
        'close': np.random.uniform(100, 200, 1000),
        'volume': np.random.randint(1000000, 10000000, 1000),
    })
    
    # Quick check
    quick_quality_check(df, "Sample Stock Data")
    
    # Monitor usage
    monitor = DataQualityMonitor()
    monitor.log_layer('bronze', df, 'after_ingestion')
    
    # Simulate Silver layer (with some filtering)
    df_silver = df[df['close'] > 120].copy()
    df_silver['daily_return'] = df_silver.groupby('ticker')['close'].pct_change()
    monitor.log_layer('silver', df_silver, 'after_cleaning')
    
    # Compare
    monitor.compare_layers('bronze', 'silver')
    
    # Summary
    print("\n📋 All Metrics:")
    print(monitor.get_summary().to_string())
