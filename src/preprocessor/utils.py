# ==========================================
# FILE: gate_preprocessor/utils.py
# ==========================================
"""Utility functions for GATE XML preprocessor."""

import logging
from pathlib import Path
from datetime import datetime
from .config import LOG_FORMAT


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_paths(organized_dir: str, output_dir: str) -> bool:
    """Validate input and output directories."""
    org_path = Path(organized_dir)
    out_path = Path(output_dir)
    
    if not org_path.exists():
        print(f"Error: Organized directory {organized_dir} does not exist")
        return False
    
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_dir}: {e}")
        return False
    
    return True


def print_processing_summary(stats: dict, tokens_count: int, docs_count: int, 
                           linguistic_features: dict) -> None:
    """Print processing summary."""
    print(f"\n🎉 GATE XML Processing complete!")
    print(f"📄 Total files: {stats['total_files']}")
    print(f"✅ Processed: {stats['processed']}")
    print(f"❌ XML errors: {stats['xml_errors']}")
    print(f"⚠️ Processing errors: {stats['processing_errors']}")
    print(f"📭 Empty files: {stats['empty_files']}")
    print(f"🔤 Total tokens: {tokens_count}")
    print(f"📚 Total documents: {docs_count}")
    
    if linguistic_features.get('spelling_variants'):
        print(f"🔄 Spelling variants: {len(linguistic_features['spelling_variants'])}")
        
    if tokens_count:
        archaic_count = sum(1 for variant in linguistic_features.get('spelling_variants', []) 
                          if variant.get('has_archaic_spelling'))
        if archaic_count:
            print(f"📜 Archaic spellings: {archaic_count} ({archaic_count/tokens_count*100:.1f}%)")