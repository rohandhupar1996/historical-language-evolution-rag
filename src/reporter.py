# ==========================================
# FILE: germanc_organizer/reporter.py
# ==========================================
"""Reporting utilities for GerManC corpus organization statistics."""

from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


class Reporter:
    """Generates reports and statistics for corpus organization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
    
    def generate_report(self, processed_files: List[Dict[str, Any]], 
                       error_files: List[str]) -> None:
        """Generate organization summary report."""
        stats = self._calculate_stats(processed_files)
        report_path = self.output_dir / "organization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("GerManC Corpus Organization Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Summary statistics
            f.write(f"Total files processed: {len(processed_files)}\n")
            f.write(f"Total files with errors: {len(error_files)}\n\n")
            
            # Files by period and genre
            f.write("Files by Period and Genre:\n")
            f.write("-" * 30 + "\n")
            
            for period in sorted(stats.keys()):
                f.write(f"\n{period}:\n")
                for genre in sorted(stats[period].keys()):
                    count = stats[period][genre]
                    f.write(f"  {genre}: {count} files\n")
            
            # Year distribution
            self._write_year_distribution(f, processed_files)
            
            # Error files
            if error_files:
                f.write("\n\nFiles with errors:\n")
                f.write("-" * 20 + "\n")
                for error_file in error_files:
                    f.write(f"  {error_file}\n")
        
        print(f"📊 Report saved to: {report_path}")
    
    def _calculate_stats(self, processed_files: List[Dict[str, Any]]) -> defaultdict:
        """Calculate statistics from processed files."""
        stats = defaultdict(lambda: defaultdict(int))
        for file_info in processed_files:
            stats[file_info['period']][file_info['genre']] += 1
        return stats
    
    def _write_year_distribution(self, f, processed_files: List[Dict[str, Any]]) -> None:
        """Write year distribution to report file."""
        f.write("\n\nYear Distribution:\n")
        f.write("-" * 20 + "\n")
        years = [info['year'] for info in processed_files]
        if years:
            f.write(f"Earliest: {min(years)}\n")
            f.write(f"Latest: {max(years)}\n")
            f.write(f"Range: {max(years) - min(years)} years\n")