# ==========================================
# FILE: preprocess.py
# ==========================================
"""Main entry point for GATE XML preprocessing."""

import argparse
from src.preprocessor import GateXMLPreprocessor
from src.preprocessor.utils import validate_paths


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Preprocess GATE XML files")
    parser.add_argument("organized_dir", help="Directory with organized XML files")
    parser.add_argument("output_dir", help="Output directory for processed data")
    
    args = parser.parse_args()
    
    # Validate paths
    if not validate_paths(args.organized_dir, args.output_dir):
        return 1
    
    # Create preprocessor and run
    preprocessor = GateXMLPreprocessor(args.organized_dir, args.output_dir)
    stats = preprocessor.process_all_files()
    
    print(f"\n📊 GATE XML Preprocessing complete!")
    print(f"Ready for Phase 3: Database creation")
    
    return 0


if __name__ == "__main__":
    exit(main())