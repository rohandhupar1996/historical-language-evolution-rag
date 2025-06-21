#!/usr/bin/env python3
"""
Run Phase 1: Organization Script
===============================

Script to run the GerManC corpus file organization phase.

This script organizes raw XML files into a structured directory hierarchy
based on time period and genre extracted from filenames.

Usage:
    python scripts/run_phase1.py --source-dir /path/to/xml/files --output-dir /path/to/organized

Example:
    python scripts/run_phase1.py \
        --source-dir /Users/rohan/Downloads/2544/LING-GATE/ \
        --output-dir /Users/rohan/Downloads/2544/organized_germanc \
        --log-level INFO
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase1_organize import GerManCOrganizer
from src.phase1_organize.utils import setup_logging, check_disk_space, find_xml_files


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check source directory
    source_path = Path(args.source_dir)
    if not source_path.exists():
        errors.append(f"Source directory does not exist: {args.source_dir}")
    elif not source_path.is_dir():
        errors.append(f"Source path is not a directory: {args.source_dir}")
    else:
        # Check for XML files
        xml_files = find_xml_files(source_path)
        if not xml_files:
            errors.append(f"No XML files found in source directory: {args.source_dir}")
        else:
            print(f"✓ Found {len(xml_files)} XML files in source directory")
    
    # Check output directory parent exists
    output_path = Path(args.output_dir)
    if not output_path.parent.exists():
        errors.append(f"Parent directory of output path does not exist: {output_path.parent}")
    
    # Check disk space
    if not check_disk_space(output_path.parent, required_space_mb=100):
        errors.append("Insufficient disk space (need at least 100MB)")
    
    return errors


def print_banner():
    """Print script banner."""
    print("🗂️  GerManC Corpus Organization - Phase 1")
    print("=" * 50)
    print("Organizing XML files by period and genre...")
    print()


def print_summary(stats, processed, errors, output_dir):
    """Print organization summary."""
    print("\n" + "=" * 50)
    print("📊 ORGANIZATION SUMMARY")
    print("=" * 50)
    
    # Overall statistics
    total_files = len(processed) + len(errors)
    success_rate = (len(processed) / total_files * 100) if total_files > 0 else 0
    
    print(f"✅ Successfully organized: {len(processed)} files")
    print(f"❌ Failed to organize: {len(errors)} files")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"📂 Output directory: {output_dir}")
    
    # Statistics by period
    if stats:
        print(f"\n📅 Distribution by Period:")
        for period, genres in stats.items():
            total = sum(genres.values())
            print(f"  {period}: {total} files")
            for genre, count in sorted(genres.items()):
                print(f"    └─ {genre}: {count} files")
    
    # Error summary
    if errors:
        print(f"\n⚠️  Files with errors:")
        for error_file in errors[:5]:  # Show first 5 errors
            print(f"    • {error_file}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
    
    print(f"\n📋 Detailed report saved in: {output_dir}/organization_report.txt")
    print(f"🔍 Metadata saved in: {output_dir}/organization_metadata.json")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Organize GerManC corpus files by period and genre',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python scripts/run_phase1.py --source-dir ./raw_xml --output-dir ./organized
  
  With custom log level:
    python scripts/run_phase1.py --source-dir ./xml --output-dir ./out --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--source-dir', 
        required=True,
        help='Directory containing raw GerManC XML files'
    )
    
    parser.add_argument(
        '--output-dir', 
        required=True,
        help='Directory where organized files will be placed'
    )
    
    parser.add_argument(
        '--log-level', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Optional log file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Validate arguments
    print("🔍 Validating arguments...")
    validation_errors = validate_arguments(args)
    
    if validation_errors:
        print("❌ Validation failed:")
        for error in validation_errors:
            print(f"   • {error}")
        return 1
    
    print("✅ Arguments validated successfully")
    
    # Handle dry run
    if args.dry_run:
        print("\n🧪 DRY RUN MODE - No files will be copied")
        # TODO: Implement dry run logic
        print("Dry run functionality would be implemented here")
        return 0
    
    try:
        print(f"\n📁 Source: {args.source_dir}")
        print(f"📁 Output: {args.output_dir}")
        print()
        
        # Create and run organizer
        organizer = GerManCOrganizer(args.source_dir, args.output_dir)
        stats, processed, errors = organizer.organize_files()
        
        # Print summary
        print_summary(stats, processed, errors, args.output_dir)
        
        # Cleanup empty directories
        print("\n🧹 Cleaning up...")
        organizer.cleanup()
        
        # Determine exit code
        if errors:
            print(f"\n⚠️  Completed with {len(errors)} errors")
            return 1
        else:
            print(f"\n🎉 All files organized successfully!")
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 130
        
    except Exception as e:
        logger.error("Unexpected error during organization: %s", e)
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)