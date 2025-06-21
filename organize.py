# ==========================================
# FILE: organize.py (Updated)
# ==========================================
"""Main entry point for GerManC corpus organizer."""

import argparse
from pathlib import Path

from germanc_organizer import GerManCOrganizer
from germanc_organizer.utils import validate_paths, print_summary


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Organize GerManC corpus files")
    parser.add_argument("source_dir", help="Source directory containing XML files")
    parser.add_argument("output_dir", help="Output directory for organized files")
    
    args = parser.parse_args()
    
    # Validate paths
    if not validate_paths(args.source_dir, args.output_dir):
        return 1
    
    # Create organizer and run
    organizer = GerManCOrganizer(args.source_dir, args.output_dir)
    stats, processed, errors = organizer.organize_files()
    
    # Print summary
    print_summary(stats, len(processed), len(errors))
    
    return 0


if __name__ == "__main__":
    exit(main())