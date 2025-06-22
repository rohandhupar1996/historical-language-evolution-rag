# ==========================================
# FILE: prepare_pipeline/utils.py
# ==========================================
"""Utility functions for prepare pipeline."""

import json
from pathlib import Path
from typing import Any, Dict
import logging


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON with proper encoding."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Path) -> Dict:
    """Load JSON data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_phase_summary(chunks_count: int, output_dir: Path) -> None:
    """Print phase completion summary."""
    print(f"\n✅ PREPARE phase completed!")
    print(f"📊 {chunks_count} chunks created")
    print(f"📁 Output: {output_dir}")
    print(f"📋 Ready for Phase 4: ACCESS")