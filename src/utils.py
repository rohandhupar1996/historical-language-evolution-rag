#!/usr/bin/env python3
"""
Utility Functions for Phase 1 Organization
==========================================

Common utility functions used across the organization modules.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union
import xml.etree.ElementTree as ET


def validate_xml_file(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file is a well-formed XML file.
    
    Args:
        file_path: Path to the XML file
        
    Returns:
        True if valid XML, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists and is not empty
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
        
        # Try to parse XML
        ET.parse(file_path)
        return True
        
    except (ET.ParseError, OSError, Exception):
        return False


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except (OSError, Exception):
        return 0


def find_xml_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    Find all XML files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of Path objects for XML files
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    if recursive:
        pattern = "**/*.xml"
    else:
        pattern = "*.xml"
    
    return list(directory.glob(pattern))


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(filename: str) -> str:
    """
    Make a filename safe for filesystem use.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename with invalid characters removed/replaced
    """
    # Characters that are problematic in filenames
    invalid_chars = '<>:"/\\|?*'
    
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove any trailing dots or spaces
    safe_name = safe_name.rstrip('. ')
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = "unnamed_file"
    
    return safe_name


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Human-readable file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def check_disk_space(directory: Union[str, Path], required_space_mb: int = 100) -> bool:
    """
    Check if there's enough disk space in the target directory.
    
    Args:
        directory: Directory to check
        required_space_mb: Required space in megabytes
        
    Returns:
        True if enough space available, False otherwise
    """
    try:
        directory = Path(directory)
        
        # Get disk usage statistics
        stat = os.statvfs(directory)
        
        # Calculate available space in MB
        available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        
        return available_mb >= required_space_mb
        
    except (OSError, AttributeError):
        # os.statvfs not available on Windows, assume space is available
        return True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_files_by_extension(directory: Union[str, Path]) -> dict:
    """
    Count files by extension in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dictionary with extensions as keys and counts as values
    """
    directory = Path(directory)
    
    if not directory.exists():
        return {}
    
    extension_counts = {}
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if not ext:
                ext = "no_extension"
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    return extension_counts


def validate_directory_structure(base_dir: Union[str, Path], expected_dirs: List[str]) -> bool:
    """
    Validate that expected directory structure exists.
    
    Args:
        base_dir: Base directory to check
        expected_dirs: List of expected subdirectory names
        
    Returns:
        True if all expected directories exist, False otherwise
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return False
    
    for expected_dir in expected_dirs:
        if not (base_dir / expected_dir).is_dir():
            return False
    
    return True


class ProgressTracker:
    """Simple progress tracker for file operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter."""
        self.current += increment
        if self.current % 10 == 0 or self.current == self.total:
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        self.logger.info("%s: %d/%d (%.1f%%)", 
                        self.description, self.current, self.total, percentage)
    
    def finish(self) -> None:
        """Mark processing as finished."""
        self.current = self.total
        self._log_progress()


def main():
    """Example usage of utility functions."""
    import tempfile
    
    print("Testing Phase 1 Utility Functions")
    print("=" * 40)
    
    # Test XML validation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write('<?xml version="1.0"?><root><item>test</item></root>')
        temp_xml = f.name
    
    print(f"XML validation test: {validate_xml_file(temp_xml)}")
    
    # Test file size
    size = get_file_size(temp_xml)
    print(f"File size: {size} bytes ({format_file_size(size)})")
    
    # Test safe filename
    unsafe_name = 'test<file>name?.xml'
    safe_name = safe_filename(unsafe_name)
    print(f"Safe filename: '{unsafe_name}' -> '{safe_name}'")
    
    # Test progress tracker
    tracker = ProgressTracker(5, "Test processing")
    for i in range(5):
        tracker.update()
    tracker.finish()
    
    # Cleanup
    os.unlink(temp_xml)
    print("\nUtility tests completed!")


if __name__ == "__main__":
    main()