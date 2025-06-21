"""Configuration settings for GerManC corpus organizer."""

from typing import Dict

# Genre mapping from filename prefixes
GENRES: Dict[str, str] = {
    'DRAM': 'Drama',
    'HUMA': 'Humanities', 
    'LEGA': 'Legal',
    'NARR': 'Narrative',
    'NEWS': 'Newspapers',
    'SCIE': 'Scientific',
    'SERM': 'Sermons'
}

# Time period bins
PERIODS: Dict[str, str] = {
    'P1': '1650-1700',
    'P2': '1700-1750', 
    'P3': '1750-1800'
}

# File pattern for parsing
FILE_PATTERN = r'([A-Z]{4})_([P][1-3])_([A-Za-z]+)_(\d{4})_(.+)\.xml'

# Default directories
DEFAULT_CONFIG = {
    'source_dir': './data/raw',
    'output_dir': './data/organized',
    'encoding': 'utf-8',
    'file_extension': '.xml'
}