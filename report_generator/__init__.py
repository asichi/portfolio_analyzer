"""
report_generator package
-------------------------
Modular HTML report generation for trading strategy analysis.

Main entry points:
- process_folder(): Process a single strategy folder
- process_all_strategies(): Process all strategies combined
"""

from .processor import process_folder, process_all_strategies

__all__ = ["process_folder", "process_all_strategies"]
