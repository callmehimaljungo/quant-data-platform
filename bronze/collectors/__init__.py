"""
Collectors Package
Multi-source data collection with automatic fallback
"""

from .base_collector import BaseCollector, RateLimitError, DataCollectionError

__all__ = ['BaseCollector', 'RateLimitError', 'DataCollectionError']
