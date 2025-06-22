# ==========================================
# FILE: prepare.py
# ==========================================
"""Main entry point for prepare phase."""

import argparse
from pathlib import Path
from prepare_pipeline import GerManCPrepareProcessor


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Prepare GerManC data for PostgreSQL")
    parser.add_argument("input_dir", help="Directory with processed data")
    parser.add_argument("output_dir", help="Output directory for prepared data")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size (default: 800)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    # Create processor and run
    processor = GerManCPrepareProcessor(args.input_dir, args.output_dir)
    chunks, features_df, stats = processor.run_prepare_phase(args.chunk_size)
    
    return 0


if __name__ == "__main__":
    exit(main())
