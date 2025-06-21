# ==========================================
# FILE: germanc_organizer/file_parser.py
# ==========================================
"""File parsing utilities for GerManC corpus files."""

import re
from typing import Optional, Dict, Any

from src.config import GENRES, PERIODS, FILE_PATTERN


class FileParser:
    """Parser for extracting metadata from GerManC corpus filenames."""
    
    def __init__(self):
        self.genres = GENRES
        self.periods = PERIODS
        self.pattern = FILE_PATTERN
    
    def extract_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extract genre, period, and year from filename.
        
        Args:
            filename: Name of the file to parse
            
        Returns:
            Dictionary with extracted info or None if parsing fails
        """
        match = re.match(self.pattern, filename)
        
        if not match:
            return None
            
        genre_code, period_code, region, year, title = match.groups()
        
        return {
            'genre': self.genres.get(genre_code, 'Unknown'),
            'genre_code': genre_code,
            'period': self.periods.get(period_code, 'Unknown'),
            'period_code': period_code,
            'region': region,
            'year': int(year),
            'title': title,
            'filename': filename
        }
    
    def is_valid_file(self, filename: str) -> bool:
        """Check if filename matches expected pattern."""
        return bool(re.match(self.pattern, filename))