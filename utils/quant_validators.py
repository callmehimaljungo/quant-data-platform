"""
Quant Dashboard - Data Validation & Formatting Utilities
=========================================================
Logic từ file tham khảo quant_validators.py để xử lý dữ liệu bất thường.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# 1. DATA VALIDATION LAYER
# =============================================================================

class ValidationSeverity(Enum):
    ERROR = "error"      # Dữ liệu không hợp lệ, cần reject
    WARNING = "warning"  # Dữ liệu đáng ngờ, cần review
    INFO = "info"        # Thông tin tham khảo


@dataclass
class ValidationResult:
    is_valid: bool
    severity: ValidationSeverity
    column: str
    message: str
    invalid_count: int
    sample_values: List[Any]


class FinancialDataValidator:
    """
    Validator cho dữ liệu tài chính.
    """
    
    # Định nghĩa ranges hợp lệ cho từng loại dữ liệu
    VALID_RANGES = {
        # Giá cổ phiếu: từ $0.01 đến $1,000,000
        'price': (0.01, 1_000_000),
        'first_price': (0.01, 1_000_000),
        'last_price': (0.01, 1_000_000),
        
        # Daily return: -100% đến +100%
        'daily_return': (-1.0, 1.0),
        'avg_daily_return': (-0.5, 0.5),
        
        # Sharpe ratio: thường từ -5 đến 10
        'sharpe_ratio': (-5.0, 10.0),
        
        # Volatility: 0% đến 500% (annualized)
        'volatility': (0, 500),
        
        # Max drawdown: luôn âm, từ -100% đến 0%
        'max_drawdown': (-100, 0),
        
        # Volume: phải dương
        'volume': (0, float('inf')),
        'avg_volume': (0, float('inf')),
    }
    
    MUST_BE_POSITIVE = ['price', 'first_price', 'last_price', 'volume', 'avg_volume', 'volatility']
    MUST_BE_NEGATIVE_OR_ZERO = ['max_drawdown']

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Returns dict keys: 'is_valid', 'errors', 'warnings', 'summary'"""
        self.results = []
        
        # Check ranges
        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col in df.columns:
                self._validate_range(df, col, min_val, max_val)
        
        # Check positive
        for col in self.MUST_BE_POSITIVE:
            if col in df.columns:
                self._validate_positive(df, col)
        
        # Check negative
        for col in self.MUST_BE_NEGATIVE_OR_ZERO:
            if col in df.columns:
                self._validate_negative_or_zero(df, col)
        
        # Check NaN/Inf
        self._validate_no_nan_inf(df)
        
        errors = [r for r in self.results if r.severity == ValidationSeverity.ERROR]
        warnings = [r for r in self.results if r.severity == ValidationSeverity.WARNING]
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'summary': {
                'total_issues': len(self.results),
                'error_count': len(errors),
                'warning_count': len(warnings),
                'rows_checked': len(df)
            }
        }
    
    def _validate_range(self, df: pd.DataFrame, col: str, min_val: float, max_val: float):
        invalid_mask = (df[col] < min_val) | (df[col] > max_val)
        invalid_df = df[invalid_mask]
        if len(invalid_df) > 0:
            self.results.append(ValidationResult(
                is_valid=False, severity=ValidationSeverity.ERROR, column=col,
                message=f"Giá trị ngoài range [{min_val}, {max_val}]",
                invalid_count=len(invalid_df), sample_values=invalid_df[col].head(5).tolist()
            ))
    
    def _validate_positive(self, df: pd.DataFrame, col: str):
        # Allow 0 for volatility sometimes, but usually >0
        invalid_mask = df[col] < 0 
        invalid_df = df[invalid_mask]
        if len(invalid_df) > 0:
            self.results.append(ValidationResult(
                is_valid=False, severity=ValidationSeverity.ERROR, column=col,
                message="Giá trị phải dương (>= 0)",
                invalid_count=len(invalid_df), sample_values=invalid_df[col].head(5).tolist()
            ))
    
    def _validate_negative_or_zero(self, df: pd.DataFrame, col: str):
        invalid_mask = df[col] > 0
        invalid_df = df[invalid_mask]
        if len(invalid_df) > 0:
            self.results.append(ValidationResult(
                is_valid=False, severity=ValidationSeverity.ERROR, column=col,
                message="Giá trị phải âm hoặc bằng 0 (<= 0)",
                invalid_count=len(invalid_df), sample_values=invalid_df[col].head(5).tolist()
            ))
    
    def _validate_no_nan_inf(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
            
            if nan_count > 0:
                self.results.append(ValidationResult(
                    is_valid=False, severity=ValidationSeverity.WARNING, column=col,
                    message=f"Có {nan_count} giá trị NaN",
                    invalid_count=nan_count, sample_values=[]
                ))
            if inf_count > 0:
                self.results.append(ValidationResult(
                    is_valid=False, severity=ValidationSeverity.ERROR, column=col,
                    message=f"Có {inf_count} giá trị Infinity",
                    invalid_count=inf_count, sample_values=[]
                ))


# =============================================================================
# 2. DATA CLEANING UTILITIES
# =============================================================================

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean và chuẩn hóa dữ liệu tài chính.
    """
    if df is None or df.empty: return df
    df = df.copy()
    
    # 1. Replace inf với NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Fix price columns (phải > 0.0001, lọc nhiễu 0.00006)
    # Nếu giá quá nhỏ (< 0.01), có thể gán NaN hoặc giữ nguyên nếu là penny stock thực sự.
    # Ở đây ta giả định < 0.001 là nhiễu data.
    price_cols = ['price', 'first_price', 'last_price']
    for col in price_cols:
        if col in df.columns:
            df.loc[df[col] < 0.001, col] = np.nan
    
    # 3. Fix max_drawdown (phải <= 0)
    if 'max_drawdown' in df.columns:
        # Nếu là số dương (ví dụ 0.2), đổi thành -0.2 (20% drawdown)
        # Nếu nó đang là percentage dạng 20.5 -> đổi thành -20.5
        mask_positive = df['max_drawdown'] > 0
        df.loc[mask_positive, 'max_drawdown'] = -df.loc[mask_positive, 'max_drawdown']
    
    # 4. Normalize daily returns (clamp to reasonable range [-1, 1] i.e. -100% to 100%)
    # Lưu ý: Return có thể lưu dưới dạng 0.01 hoặc 1.0 (percent). Cần check magnitude.
    # Trong project này, return thường là decimal (0.01), volatility là percent (28.5).
    
    # Xử lý avg_daily_return cực lớn (67450897)
    if 'avg_daily_return' in df.columns:
        # Nếu > 10 (1000% daily), chắc chắn là lỗi -> NaN
        df.loc[df['avg_daily_return'].abs() > 10, 'avg_daily_return'] = np.nan

    return df
