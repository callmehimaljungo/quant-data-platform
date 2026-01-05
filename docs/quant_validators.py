"""
Quant Dashboard - Data Validation & Formatting Utilities
=========================================================
Các hàm này giúp fix các lỗi Critical và High priority đã được phát hiện.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings


# =============================================================================
# 1. DATA VALIDATION LAYER (Fix C1, C3)
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
    Sử dụng: 
        validator = FinancialDataValidator()
        results = validator.validate(df)
        if not results['is_valid']:
            for error in results['errors']:
                print(f"Error in {error.column}: {error.message}")
    """
    
    # Định nghĩa ranges hợp lệ cho từng loại dữ liệu
    VALID_RANGES = {
        # Giá cổ phiếu: từ $0.01 đến $1,000,000 (Berkshire Hathaway Class A ~ $500k)
        'price': (0.01, 1_000_000),
        'first_price': (0.01, 1_000_000),
        'last_price': (0.01, 1_000_000),
        
        # Daily return: -100% đến +100% (penny stocks có thể dao động mạnh)
        'daily_return': (-1.0, 1.0),
        'avg_daily_return': (-0.5, 0.5),  # Trung bình không thể quá extreme
        
        # Sharpe ratio: thường từ -3 đến 5 (>3 đã rất exceptional)
        'sharpe_ratio': (-5.0, 10.0),
        
        # Volatility: 0% đến 500% (annualized)
        'volatility': (0, 500),
        
        # Max drawdown: luôn âm, từ -100% đến 0%
        'max_drawdown': (-100, 0),
        
        # Price change percent: -99% đến +10000% (cho penny stocks)
        'price_change_pct': (-99, 10000),
        
        # Volume: phải dương
        'volume': (0, float('inf')),
        'avg_volume': (0, float('inf')),
    }
    
    # Các cột phải có giá trị dương
    MUST_BE_POSITIVE = ['price', 'first_price', 'last_price', 'volume', 'avg_volume', 'volatility']
    
    # Các cột phải có giá trị âm hoặc 0
    MUST_BE_NEGATIVE_OR_ZERO = ['max_drawdown']

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: Nếu True, sẽ raise exception khi gặp lỗi nghiêm trọng
        """
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate toàn bộ DataFrame.
        
        Returns:
            Dict với keys: 'is_valid', 'errors', 'warnings', 'summary'
        """
        self.results = []
        
        # Check từng cột có trong VALID_RANGES
        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col in df.columns:
                self._validate_range(df, col, min_val, max_val)
        
        # Check các cột phải dương
        for col in self.MUST_BE_POSITIVE:
            if col in df.columns:
                self._validate_positive(df, col)
        
        # Check các cột phải âm
        for col in self.MUST_BE_NEGATIVE_OR_ZERO:
            if col in df.columns:
                self._validate_negative_or_zero(df, col)
        
        # Check for NaN/Inf
        self._validate_no_nan_inf(df)
        
        # Check for duplicate format issues (như "4596299745.3.7292")
        self._validate_numeric_format(df)
        
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
        """Kiểm tra giá trị trong range hợp lệ."""
        invalid_mask = (df[col] < min_val) | (df[col] > max_val)
        invalid_df = df[invalid_mask]
        
        if len(invalid_df) > 0:
            self.results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                column=col,
                message=f"Giá trị ngoài range [{min_val}, {max_val}]",
                invalid_count=len(invalid_df),
                sample_values=invalid_df[col].head(5).tolist()
            ))
    
    def _validate_positive(self, df: pd.DataFrame, col: str):
        """Kiểm tra giá trị phải dương."""
        invalid_mask = df[col] <= 0
        invalid_df = df[invalid_mask]
        
        if len(invalid_df) > 0:
            self.results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                column=col,
                message="Giá trị phải dương (> 0)",
                invalid_count=len(invalid_df),
                sample_values=invalid_df[col].head(5).tolist()
            ))
    
    def _validate_negative_or_zero(self, df: pd.DataFrame, col: str):
        """Kiểm tra giá trị phải âm hoặc bằng 0."""
        invalid_mask = df[col] > 0
        invalid_df = df[invalid_mask]
        
        if len(invalid_df) > 0:
            self.results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                column=col,
                message="Giá trị phải âm hoặc bằng 0 (≤ 0)",
                invalid_count=len(invalid_df),
                sample_values=invalid_df[col].head(5).tolist()
            ))
    
    def _validate_no_nan_inf(self, df: pd.DataFrame):
        """Kiểm tra không có NaN hoặc Inf."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
            
            if nan_count > 0:
                self.results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    message=f"Có {nan_count} giá trị NaN",
                    invalid_count=nan_count,
                    sample_values=[]
                ))
            
            if inf_count > 0:
                self.results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    column=col,
                    message=f"Có {inf_count} giá trị Infinity",
                    invalid_count=inf_count,
                    sample_values=[]
                ))
    
    def _validate_numeric_format(self, df: pd.DataFrame):
        """Kiểm tra format số (detect issues như multiple decimal points)."""
        # Chỉ check nếu data ban đầu là string
        object_cols = df.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            # Check for multiple decimal points
            if df[col].dtype == object:
                invalid_mask = df[col].astype(str).str.count(r'\.') > 1
                invalid_df = df[invalid_mask]
                
                if len(invalid_df) > 0:
                    self.results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        message="Giá trị có nhiều hơn 1 dấu chấm (format error)",
                        invalid_count=len(invalid_df),
                        sample_values=invalid_df[col].head(5).tolist()
                    ))


# =============================================================================
# 2. NUMBER FORMATTING UTILITIES (Fix H3, M8)
# =============================================================================

class NumberFormatter:
    """
    Formatter cho các loại số khác nhau trong dashboard tài chính.
    
    Sử dụng:
        formatter = NumberFormatter()
        print(formatter.price(1234.5678))  # "1,234.57"
        print(formatter.percent(0.1234))   # "12.34%"
    """
    
    @staticmethod
    def price(value: Optional[float], decimals: int = 2) -> str:
        """Format giá tiền: $1,234.56"""
        if value is None or pd.isna(value):
            return "-"
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def percent(value: Optional[float], decimals: int = 2, multiply: bool = True) -> str:
        """
        Format phần trăm.
        Args:
            value: Giá trị (0.1234 hoặc 12.34 tùy multiply)
            decimals: Số chữ số thập phân
            multiply: Nếu True, nhân với 100 (0.1234 -> 12.34%)
        """
        if value is None or pd.isna(value):
            return "-"
        display_value = value * 100 if multiply else value
        return f"{display_value:.{decimals}f}%"
    
    @staticmethod
    def ratio(value: Optional[float], decimals: int = 4) -> str:
        """Format tỷ lệ: 0.1234"""
        if value is None or pd.isna(value):
            return "-"
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def volume(value: Optional[float]) -> str:
        """Format volume với suffix K/M/B."""
        if value is None or pd.isna(value):
            return "-"
        if value >= 1e9:
            return f"{value / 1e9:.2f}B"
        if value >= 1e6:
            return f"{value / 1e6:.2f}M"
        if value >= 1e3:
            return f"{value / 1e3:.2f}K"
        return f"{value:,.0f}"
    
    @staticmethod
    def sharpe(value: Optional[float]) -> str:
        """Format Sharpe ratio: 2 decimals."""
        if value is None or pd.isna(value):
            return "-"
        return f"{value:.2f}"
    
    @staticmethod
    def integer(value: Optional[float]) -> str:
        """Format số nguyên với comma separator."""
        if value is None or pd.isna(value):
            return "-"
        return f"{int(value):,}"


# =============================================================================
# 3. DATA CLEANING UTILITIES
# =============================================================================

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean và chuẩn hóa dữ liệu tài chính.
    
    Các bước:
    1. Handle missing values
    2. Fix negative prices
    3. Normalize percentages
    4. Remove outliers (optional)
    """
    df = df.copy()
    
    # 1. Replace inf với NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Fix price columns (phải > 0)
    price_cols = ['price', 'first_price', 'last_price']
    for col in price_cols:
        if col in df.columns:
            # Replace invalid prices với NaN
            df.loc[df[col] <= 0, col] = np.nan
    
    # 3. Fix max_drawdown (phải <= 0)
    if 'max_drawdown' in df.columns:
        # Nếu là số dương, đổi dấu
        df.loc[df['max_drawdown'] > 0, 'max_drawdown'] *= -1
    
    # 4. Normalize daily returns (clamp to reasonable range)
    return_cols = ['daily_return', 'avg_daily_return']
    for col in return_cols:
        if col in df.columns:
            df[col] = df[col].clip(-1, 1)
    
    return df


def adjust_for_stock_splits(df: pd.DataFrame, splits: Dict[str, List[tuple]]) -> pd.DataFrame:
    """
    Điều chỉnh giá cho stock splits.
    
    Args:
        df: DataFrame với columns ['ticker', 'date', 'price']
        splits: Dict mapping ticker -> list of (date, ratio) tuples
                Example: {'AAPL': [('2020-08-31', 4), ('2014-06-09', 7)]}
    
    Returns:
        DataFrame với giá đã điều chỉnh
    """
    df = df.copy()
    
    for ticker, split_list in splits.items():
        ticker_mask = df['ticker'] == ticker
        
        for split_date, ratio in split_list:
            # Điều chỉnh giá trước ngày split
            date_mask = df['date'] < split_date
            df.loc[ticker_mask & date_mask, 'price'] /= ratio
    
    return df


# =============================================================================
# 4. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Tạo sample data với các lỗi như trong screenshots
    sample_data = pd.DataFrame({
        'ticker': ['BF-A', 'MCD', 'ETN', 'MDU', 'WMT'],
        'sector': ['Consumer Defensive', 'Consumer Cyclical', 'Industrials', 'Utilities', 'Consumer Defensive'],
        'first_price': [0.00006, 0.0053, 0.0103, 0.00000003, 0.0038],  # Lỗi: giá quá thấp
        'last_price': [43.05, 293.5, 334.21, 15.59, 82.82],
        'avg_daily_return': [67450897.2346, 5556994.1379, 3252492.5038, 4596299745.3, 2184059.9611],  # Lỗi: quá lớn
        'volatility': [26.7991, 28.5204, 30.1795, 70.9624, 29.2872],
        'sharpe_ratio': [1.0289, 0.7687, 0.7363, 0.7331, 0.7313],
        'max_drawdown': [-69.11, -73.6312, -67.3184, -61.3822, -76.7139]
    })
    
    print("=" * 60)
    print("VALIDATION DEMO")
    print("=" * 60)
    
    # Validate
    validator = FinancialDataValidator()
    results = validator.validate(sample_data)
    
    print(f"\n✅ Is Valid: {results['is_valid']}")
    print(f"📊 Summary: {results['summary']}")
    
    if results['errors']:
        print("\n❌ ERRORS:")
        for error in results['errors']:
            print(f"  - {error.column}: {error.message}")
            print(f"    Invalid count: {error.invalid_count}")
            print(f"    Samples: {error.sample_values}")
    
    if results['warnings']:
        print("\n⚠️ WARNINGS:")
        for warning in results['warnings']:
            print(f"  - {warning.column}: {warning.message}")
    
    print("\n" + "=" * 60)
    print("FORMATTING DEMO")
    print("=" * 60)
    
    formatter = NumberFormatter()
    print(f"\nPrice: {formatter.price(1234567.89)}")
    print(f"Percent: {formatter.percent(0.1234)}")
    print(f"Ratio: {formatter.ratio(0.123456789)}")
    print(f"Volume: {formatter.volume(1234567890)}")
    print(f"Sharpe: {formatter.sharpe(1.2345)}")
