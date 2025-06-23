# ==========================================
# FILE: main.py
# ==========================================
"""Main entry point for GerManC corpus organizer."""

import argparse
from pathlib import Path

from src.organizer import GerManCOrganizer
from utils.utils import validate_paths, print_summary


def main():
    """Main function with CLI interface."""
    print(1)

    parser = argparse.ArgumentParser(description="Organize GerManC corpus files")
    parser.add_argument("source_dir", help="Source directory containing XML files")
    parser.add_argument("output_dir", help="Output directory for organized files")
    print(1)

    args = parser.parse_args()
    
    # Validate paths
    if not validate_paths(args.source_dir, args.output_dir):
        return 1
    
    # Create organizer and run
    print(1)
    organizer = GerManCOrganizer(args.source_dir, args.output_dir)
    stats, processed, errors = organizer.organize_files()
    
    # Print summary
    print_summary(stats, len(processed), len(errors))
    
    return 0


if __name__ == "__main__":
    exit(main())
