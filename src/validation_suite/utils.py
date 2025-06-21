# ==========================================
# FILE: validation_suite/utils.py
# ==========================================
"""Validation utilities."""

import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def print_validation_summary(critical_errors: list, warnings: list, 
                           validation_results: dict) -> None:
    """Print validation summary."""
    print("\n" + "="*60)
    print("🎯 VALIDATION SUMMARY")
    print("="*60)
    
    if critical_errors:
        print("❌ CRITICAL ERRORS:")
        for error in critical_errors:
            print(f"   • {error}")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not critical_errors:
        print("✅ NO CRITICAL ERRORS - Data ready for RAG pipeline!")
    else:
        print(f"\n❌ Fix {len(critical_errors)} critical errors before proceeding")