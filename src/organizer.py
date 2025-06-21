#!/usr/bin/env python3
"""
Main GerManC Organizer
=====================

This is the main organizer class that orchestrates the organization of
GerManC XML files into a structured directory hierarchy.

Usage:
    from src.phase1_organize import GerManCOrganizer
    
    organizer = GerManCOrganizer(source_dir, output_dir)
    stats, processed, errors = organizer.organize_files()
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from .file_info_extractor import FileInfoExtractor
from .directory_manager import DirectoryManager
from .report_generator import ReportGenerator


class GerManCOrganizer:
    """
    Main organizer class for GerManC corpus files.
    
    Organizes XML files by period and genre based on filename patterns.
    """
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize the organizer.
        
        Args:
            source_dir: Directory containing raw XML files
            output_dir: Directory to create organized structure
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Validate source directory
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        if not self.source_dir.is_dir():
            raise ValueError(f"Source path is not a directory: {source_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.file_extractor = FileInfoExtractor()
        self.directory_manager = DirectoryManager(
            str(self.output_dir), 
            self.file_extractor.get_supported_genres(),
            self.file_extractor.get_supported_periods()
        )
        self.report_generator = ReportGenerator(str(self.output_dir))
        
        self.logger.info("GerManCOrganizer initialized: %s -> %s", source_dir, output_dir)
    
    def organize_files(self) -> Tuple[Dict[str, Dict[str, int]], List[Dict], List[str]]:
        """
        Main method to organize all XML files.
        
        Returns:
            Tuple of (statistics, processed_files, error_files)
        """
        self.logger.info("Starting file organization...")
        
        # Create directory structure
        self.directory_manager.create_directory_structure()
        
        # Initialize tracking
        stats = defaultdict(lambda: defaultdict(int))
        processed_files = []
        error_files = []
        
        # Find all XML files
        xml_files = list(self.source_dir.glob("*.xml"))
        self.logger.info("Found %d XML files to process", len(xml_files))
        
        if not xml_files:
            self.logger.warning("No XML files found in %s", self.source_dir)
            return dict(stats), processed_files, error_files
        
        # Process each XML file
        for xml_file in xml_files:
            try:
                success = self._process_single_file(xml_file, stats, processed_files, error_files)
                if success:
                    self.logger.debug("✓ Successfully processed: %s", xml_file.name)
                else:
                    self.logger.debug("✗ Failed to process: %s", xml_file.name)
                    
            except Exception as e:
                error_files.append(xml_file.name)
                self.logger.error("Unexpected error processing %s: %s", xml_file.name, e)
        
        # Generate reports
        self._generate_reports(stats, processed_files, error_files)
        
        self.logger.info("Organization complete: %d processed, %d errors", 
                        len(processed_files), len(error_files))
        
        return dict(stats), processed_files, error_files
    
    def _process_single_file(
        self, 
        xml_file: Path, 
        stats: Dict, 
        processed_files: List[Dict], 
        error_files: List[str]
    ) -> bool:
        """
        Process a single XML file.
        
        Args:
            xml_file: Path to the XML file
            stats: Statistics dictionary to update
            processed_files: List to append successful files
            error_files: List to append failed files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract file information
            file_info = self.file_extractor.extract_file_info(xml_file.name)
            
            if not file_info:
                error_files.append(xml_file.name)
                return False
            
            # Validate file info
            if not self.file_extractor.validate_file_info(file_info):
                error_files.append(xml_file.name)
                self.logger.warning("Invalid file info for: %s", xml_file.name)
                return False
            
            # Copy file to organized location
            dest_path = self.directory_manager.copy_file(xml_file, file_info)
            
            # Add additional metadata
            file_info.update({
                'source_path': str(xml_file),
                'dest_path': str(dest_path),
                'file_size': xml_file.stat().st_size
            })
            
            # Update statistics
            stats[file_info['period']][file_info['genre']] += 1
            processed_files.append(file_info)
            
            print(f"✓ {xml_file.name} → {file_info['period']}/{file_info['genre']}")
            
            return True
            
        except Exception as e:
            error_files.append(xml_file.name)
            self.logger.error("Error processing %s: %s", xml_file.name, e)
            return False
    
    def _generate_reports(
        self, 
        stats: Dict, 
        processed_files: List[Dict], 
        error_files: List[str]
    ) -> None:
        """Generate organization reports."""
        try:
            # Generate main organization report
            report_path = self.report_generator.generate_organization_report(
                stats, processed_files, error_files
            )
            
            # Generate JSON metadata
            self._save_metadata(processed_files, stats)
            
            print(f"\n📊 Report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error("Error generating reports: %s", e)
    
    def _save_metadata(self, processed_files: List[Dict], stats: Dict) -> None:
        """Save metadata as JSON for later use."""
        import json
        from datetime import datetime
        
        metadata = {
            'organization_timestamp': datetime.now().isoformat(),
            'total_files': len(processed_files),
            'statistics': dict(stats),
            'supported_genres': self.file_extractor.get_supported_genres(),
            'supported_periods': self.file_extractor.get_supported_periods(),
            'files': processed_files
        }
        
        metadata_path = self.output_dir / "organization_metadata.json"
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Metadata saved to: %s", metadata_path)
            
        except Exception as e:
            self.logger.error("Error saving metadata: %s", e)
    
    def get_organization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the organization results.
        
        Returns:
            Dictionary with organization summary
        """
        metadata_path = self.output_dir / "organization_metadata.json"
        
        if not metadata_path.exists():
            return {"error": "No organization metadata found"}
        
        try:
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                'total_files': metadata.get('total_files', 0),
                'organization_date': metadata.get('organization_timestamp'),
                'statistics': metadata.get('statistics', {}),
                'output_directory': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error("Error loading organization summary: %s", e)
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up empty directories and temporary files."""
        try:
            removed = self.directory_manager.cleanup_empty_directories()
            if removed > 0:
                self.logger.info("Cleaned up %d empty directories", removed)
        except Exception as e:
            self.logger.error("Error during cleanup: %s", e)


def main():
    """Example usage of GerManCOrganizer."""
    import argparse
    import sys
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Organize GerManC corpus files')
    parser.add_argument('--source-dir', required=True, 
                       help='Source directory with XML files')
    parser.add_argument('--output-dir', required=True, 
                       help='Output directory for organized files')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and run organizer
        organizer = GerManCOrganizer(args.source_dir, args.output_dir)
        stats, processed, errors = organizer.organize_files()
        
        # Print summary
        print(f"\n🎉 Organization complete!")
        print(f"📁 Organized {len(processed)} files")
        print(f"❌ {len(errors)} errors")
        print(f"📂 Output directory: {args.output_dir}")
        
        # Show statistics
        print(f"\n📊 Statistics by period:")
        for period, genres in stats.items():
            total = sum(genres.values())
            print(f"  {period}: {total} files")
            for genre, count in genres.items():
                print(f"    {genre}: {count}")
        
        # Cleanup
        organizer.cleanup()
        
        return 0 if not errors else 1
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())