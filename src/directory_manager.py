# ==========================================
# FILE: germanc_organizer/directory_manager.py
# ==========================================
"""Directory management utilities for GerManC corpus organization."""

import shutil
from pathlib import Path
from typing import List

from .config import GENRES, PERIODS


class DirectoryManager:
    """Manages directory structure for organized corpus files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.genres = GENRES
        self.periods = PERIODS
    
    def create_directory_structure(self) -> None:
        """Create organized directory structure."""
        for period in self.periods.values():
            for genre in self.genres.values():
                dir_path = self.output_dir / period / genre
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def copy_file(self, source_file: Path, period: str, genre: str) -> Path:
        """Copy file to organized location.
        
        Args:
            source_file: Source file path
            period: Time period for organization
            genre: Genre for organization
            
        Returns:
            Destination file path
        """
        dest_dir = self.output_dir / period / genre
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_dir / source_file.name
        shutil.copy2(source_file, dest_path)
        
        return dest_path
    
    def get_all_xml_files(self, source_dir: Path) -> List[Path]:
        """Get all XML files from source directory."""
        return list(Path(source_dir).glob("*.xml"))