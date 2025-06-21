# ==========================================
# FILE: germanc_organizer/utils.py
# ==========================================
"""Utility functions for GerManC corpus organizer."""

import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('germanc_organizer.log')
        ]
    )


def validate_paths(source_dir: str, output_dir: str) -> bool:
    """Validate that source directory exists and output directory can be created."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return False
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_dir}: {e}")
        return False
    
    return True


def print_summary(stats: Dict[str, Any], processed: int, errors: int) -> None:
    """Print organization summary."""
    print(f"\n🎉 Organization complete!")
    print(f"📁 Organized {processed} files")
    print(f"❌ {errors} errors")
    print(f"📊 Check organization_report.txt for details")