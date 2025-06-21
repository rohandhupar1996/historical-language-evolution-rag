#!/usr/bin/env python3
"""
Test Suite for Phase 1: Organization
====================================

Unit tests for the GerManC corpus organization modules.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase1_organize.file_info_extractor import FileInfoExtractor
from src.phase1_organize.directory_manager import DirectoryManager
from src.phase1_organize.organizer import GerManCOrganizer
from src.phase1_organize.utils import validate_xml_file, safe_filename, format_file_size


class TestFileInfoExtractor(unittest.TestCase):
    """Test FileInfoExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FileInfoExtractor()
    
    def test_valid_filename_parsing(self):
        """Test parsing of valid GerManC filenames."""
        filename = "DRAM_P1_NoD_1673_Leonilda.xml"
        info = self.extractor.extract_file_info(filename)
        
        self.assertIsNotNone(info)
        self.assertEqual(info['genre'], 'Drama')
        self.assertEqual(info['period'], '1650-1700')
        self.assertEqual(info['year'], 1673)
        self.assertEqual(info['title'], 'Leonilda')
        self.assertEqual(info['region'], 'NoD')
        self.assertEqual(info['filename'], filename)
    
    def test_invalid_filename_patterns(self):
        """Test handling of invalid filename patterns."""
        invalid_files = [
            "invalid_filename.xml",
            "DRAM_P1_NoD_1673.xml",  # Missing title
            "DRAM_P1_NoD_Leonilda.xml",  # Missing year
            "XXXX_P1_NoD_1673_Leonilda.xml",  # Invalid genre
            "DRAM_P9_NoD_1673_Leonilda.xml",  # Invalid period
            "not_xml_file.txt",  # Not XML
        ]
        
        for filename in invalid_files:
            with self.subTest(filename=filename):
                info = self.extractor.extract_file_info(filename)
                self.assertIsNone(info)
    
    def test_edge_cases(self):
        """Test edge cases in filename parsing."""
        # Empty filename
        self.assertIsNone(self.extractor.extract_file_info(""))
        
        # None input
        self.assertIsNone(self.extractor.extract_file_info(None))
        
        # Complex title with underscores
        filename = "HUMA_P2_WOD_1744_Complex_Title_With_Underscores.xml"
        info = self.extractor.extract_file_info(filename)
        self.assertIsNotNone(info)
        self.assertEqual(info['title'], 'Complex_Title_With_Underscores')
    
    def test_supported_mappings(self):
        """Test that supported genres and periods are returned correctly."""
        genres = self.extractor.get_supported_genres()
        periods = self.extractor.get_supported_periods()
        
        self.assertIn('DRAM', genres)
        self.assertIn('HUMA', genres)
        self.assertEqual(genres['DRAM'], 'Drama')
        
        self.assertIn('P1', periods)
        self.assertIn('P2', periods)
        self.assertEqual(periods['P1'], '1650-1700')


class TestDirectoryManager(unittest.TestCase):
    """Test DirectoryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.genres = {'DRAM': 'Drama', 'HUMA': 'Humanities'}
        self.periods = {'P1': '1650-1700', 'P2': '1700-1750'}
        self.manager = DirectoryManager(self.temp_dir, self.genres, self.periods)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_directory_creation(self):
        """Test creation of directory structure."""
        self.manager.create_directory_structure()
        
        # Check that all expected directories exist
        for period in self.periods.values():
            for genre in self.genres.values():
                expected_dir = Path(self.temp_dir) / period / genre
                self.assertTrue(expected_dir.exists())
                self.assertTrue(expected_dir.is_dir())
    
    def test_get_destination_path(self):
        """Test destination path calculation."""
        file_info = {
            'period': '1650-1700',
            'genre': 'Drama',
            'filename': 'test.xml'
        }
        
        dest_path = self.manager.get_destination_path(file_info)
        expected_path = Path(self.temp_dir) / '1650-1700' / 'Drama' / 'test.xml'
        
        self.assertEqual(dest_path, expected_path)
    
    def test_file_copy(self):
        """Test file copying functionality."""
        # Create a test source file
        source_file = Path(self.temp_dir) / "source.xml"
        source_file.write_text("<?xml version='1.0'?><test/>")
        
        file_info = {
            'period': '1650-1700',
            'genre': 'Drama',
            'filename': 'copied.xml'
        }
        
        # Copy the file
        dest_path = self.manager.copy_file(source_file, file_info)
        
        # Verify the file was copied
        self.assertTrue(dest_path.exists())
        self.assertEqual(dest_path.read_text(), source_file.read_text())
    
    def test_file_stats(self):
        """Test file statistics calculation."""
        # Create directory structure
        self.manager.create_directory_structure()
        
        # Create some test files
        drama_dir = Path(self.temp_dir) / '1650-1700' / 'Drama'
        (drama_dir / 'file1.xml').write_text("test")
        (drama_dir / 'file2.xml').write_text("test")
        
        huma_dir = Path(self.temp_dir) / '1650-1700' / 'Humanities'
        (huma_dir / 'file3.xml').write_text("test")
        
        # Get stats
        stats = self.manager.get_file_stats()
        
        self.assertEqual(stats['1650-1700']['Drama'], 2)
        self.assertEqual(stats['1650-1700']['Humanities'], 1)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_safe_filename(self):
        """Test safe filename generation."""
        test_cases = [
            ("normal_file.xml", "normal_file.xml"),
            ("file<with>bad?chars.xml", "file_with_bad_chars.xml"),
            ("file with spaces.xml", "file with spaces.xml"),
            ("", "unnamed_file"),
            ("file...", "file"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = safe_filename(original)
                self.assertEqual(result, expected)
    
    def test_format_file_size(self):
        """Test file size formatting."""
        test_cases = [
            (0, "0 B"),
            (512, "512.0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
        ]
        
        for size_bytes, expected in test_cases:
            with self.subTest(size_bytes=size_bytes):
                result = format_file_size(size_bytes)
                self.assertEqual(result, expected)
    
    def test_validate_xml_file(self):
        """Test XML file validation."""
        # Create valid XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<?xml version="1.0"?><root><item>test</item></root>')
            valid_xml = f.name
        
        # Create invalid XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<invalid><xml>')
            invalid_xml = f.name
        
        try:
            self.assertTrue(validate_xml_file(valid_xml))
            self.assertFalse(validate_xml_file(invalid_xml))
            self.assertFalse(validate_xml_file("nonexistent_file.xml"))
        finally:
            Path(valid_xml).unlink()
            Path(invalid_xml).unlink()


class TestGerManCOrganizer(unittest.TestCase):
    """Test main GerManCOrganizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_source = tempfile.mkdtemp()
        self.temp_output = tempfile.mkdtemp()
        
        # Create some test XML files
        test_files = [
            "DRAM_P1_NoD_1673_Leonilda.xml",
            "HUMA_P2_WOD_1744_Pfaltz.xml",
            "SCIE_P3_OMD_1781_Chymie.xml",
        ]
        
        for filename in test_files:
            file_path = Path(self.temp_source) / filename
            file_path.write_text('<?xml version="1.0"?><document><content>Sample content</content></document>')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_source)
        shutil.rmtree(self.temp_output)
    
    def test_organization_process(self):
        """Test the complete organization process."""
        organizer = GerManCOrganizer(self.temp_source, self.temp_output)
        stats, processed, errors = organizer.organize_files()
        
        # Check that files were processed
        self.assertEqual(len(processed), 3)
        self.assertEqual(len(errors), 0)
        
        # Check statistics
        self.assertIn('1650-1700', stats)
        self.assertIn('1700-1750', stats)
        self.assertIn('1750-1800', stats)
        
        # Check that files were actually copied
        drama_file = Path(self.temp_output) / '1650-1700' / 'Drama' / 'DRAM_P1_NoD_1673_Leonilda.xml'
        self.assertTrue(drama_file.exists())
        
        huma_file = Path(self.temp_output) / '1700-1750' / 'Humanities' / 'HUMA_P2_WOD_1744_Pfaltz.xml'
        self.assertTrue(huma_file.exists())
    
    def test_invalid_source_directory(self):
        """Test handling of invalid source directory."""
        with self.assertRaises(ValueError):
            GerManCOrganizer("/nonexistent/directory", self.temp_output)
    
    def test_empty_source_directory(self):
        """Test handling of empty source directory."""
        empty_dir = tempfile.mkdtemp()
        try:
            organizer = GerManCOrganizer(empty_dir, self.temp_output)
            stats, processed, errors = organizer.organize_files()
            
            self.assertEqual(len(processed), 0)
            self.assertEqual(len(errors), 0)
        finally:
            shutil.rmtree(empty_dir)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestFileInfoExtractor,
        TestDirectoryManager,
        TestUtils,
        TestGerManCOrganizer,
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    
    print("🧪 Running Phase 1 Organization Tests")
    print("=" * 40)
    
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(exit_code)