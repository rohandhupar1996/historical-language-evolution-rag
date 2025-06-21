# ==========================================
# FILE: germanc_organizer/organizer.py
# ==========================================
"""Main organizer class for GerManC corpus files."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.file_parser import FileParser
from src.directory_manager import DirectoryManager
from src.reporter import Reporter
from utils.utils import setup_logging


class GerManCOrganizer:
    """Main class for organizing GerManC corpus files."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        self.parser = FileParser()
        self.dir_manager = DirectoryManager(self.output_dir)
        self.reporter = Reporter(self.output_dir)
        
        setup_logging()
    
    def organize_files(self) -> Tuple[defaultdict, List[Dict[str, Any]], List[str]]:
        """Main function to organize all GerManC files."""
        # Create output directory structure
        self.dir_manager.create_directory_structure()
        
        # Initialize tracking variables
        processed_files = []
        error_files = []
        
        # Process each XML file
        xml_files = self.dir_manager.get_all_xml_files(self.source_dir)
        
        for xml_file in xml_files:
            try:
                file_info = self.parser.extract_file_info(xml_file.name)
                
                if file_info:
                    # Copy file to organized location
                    dest_path = self.dir_manager.copy_file(
                        xml_file, 
                        file_info['period'], 
                        file_info['genre']
                    )
                    
                    processed_files.append(file_info)
                    print(f"✓ {xml_file.name} → {file_info['period']}/{file_info['genre']}")
                    
                else:
                    error_files.append(xml_file.name)
                    print(f"✗ Could not parse: {xml_file.name}")
                    
            except Exception as e:
                error_files.append(xml_file.name)
                print(f"✗ Error processing {xml_file.name}: {e}")
        
        # Calculate stats
        stats = self._calculate_stats(processed_files)
        
        # Generate summary report
        self.reporter.generate_report(processed_files, error_files)
        
        return stats, processed_files, error_files
    
    def _calculate_stats(self, processed_files: List[Dict[str, Any]]) -> defaultdict:
        """Calculate organization statistics."""
        stats = defaultdict(lambda: defaultdict(int))
        for file_info in processed_files:
            stats[file_info['period']][file_info['genre']] += 1
        return stats