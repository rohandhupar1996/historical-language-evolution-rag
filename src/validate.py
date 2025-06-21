# ==========================================
# FILE: validate.py
# ==========================================
"""Main entry point for validation."""

import argparse
from validation_suite import GerManCValidator
from pathlib import Path


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate processed GerManC data")
    parser.add_argument("processed_dir", help="Directory with processed data")
    
    args = parser.parse_args()
    
    if not Path(args.processed_dir).exists():
        print(f"Error: Directory {args.processed_dir} does not exist")
        return 1
    
    validator = GerManCValidator(args.processed_dir)
    results = validator.run_full_validation()
    
    print("\n🎉 Validation Complete!")
    print("Check validation_report.json for detailed results")
    
    return 0


if __name__ == "__main__":
    exit(main())