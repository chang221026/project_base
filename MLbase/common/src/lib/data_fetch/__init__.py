"""Data fetching library.
 
Provides data loading and fetching functionality.
"""
from utils.registry import Registry
 
# Create data fetching registry
DATA_FETCHERS = Registry('data_fetcher')
 
# Import base classes
from .base import BaseDataFetcher
 
# Import implementations to trigger registration
from . import csv_fetcher
 
__all__ = ['DATA_FETCHERS', 'BaseDataFetcher']