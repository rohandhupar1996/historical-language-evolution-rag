# ==========================================
# FILE: gate_preprocessor/preprocessor.py
# ==========================================
"""Main GATE XML preprocessor class."""

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from .xml_parser import XMLParser
from .text_processor import TextProcessor
from .annotation_extractor import AnnotationExtractor
from .token_processor import TokenProcessor
from .data_saver import DataSaver
from .statistics_generator import StatisticsGenerator
from .utils import setup_logging, print_processing_summary


class GateXMLPreprocessor:
    """Main class for processing GATE XML files."""
    
    def __init__(self, organized_dir: str, output_dir: str):
        self.organized_dir = Path(organized_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.xml_parser = XMLParser()
        self.text_processor = TextProcessor()
        self.annotation_extractor = AnnotationExtractor()
        self.token_processor = TokenProcessor()
        self.data_saver = DataSaver(self.output_dir)
        self.stats_generator = StatisticsGenerator(self.output_dir)
        
        # Initialize data containers
        self.documents = []
        self.tokens = []
        self.linguistic_features = defaultdict(list)
        
        # Setup logging and stats
        self.logger = setup_logging(self.output_dir)
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'xml_errors': 0,
            'processing_errors': 0,
            'empty_files': 0
        }
    
    def process_all_files(self) -> Dict:
        """Process all organized XML files."""
        xml_files = self._collect_xml_files()
        self.stats['total_files'] = len(xml_files)
        
        # Process with progress bar
        with tqdm(xml_files, desc="Processing GATE XML files", unit="file") as pbar:
            for xml_file, period, genre in pbar:
                pbar.set_postfix(file=xml_file.name[:30])
                self._process_single_file(xml_file, period, genre)
        
        # Save all data
        self.data_saver.save_all_data(self.documents, self.tokens, self.linguistic_features)
        self.stats_generator.generate_and_save_statistics(
            self.documents, self.tokens, self.linguistic_features, self.stats
        )
        
        # Print summary
        print_processing_summary(self.stats, len(self.tokens), len(self.documents), self.linguistic_features)
        
        return self.stats
    
    def _collect_xml_files(self) -> List[Tuple[Path, str, str]]:
        """Collect all XML files from organized directory structure."""
        xml_files = []
        for period_dir in self.organized_dir.iterdir():
            if period_dir.is_dir():
                period = period_dir.name
                for genre_dir in period_dir.iterdir():
                    if genre_dir.is_dir():
                        genre = genre_dir.name
                        for xml_file in genre_dir.glob("*.xml"):
                            xml_files.append((xml_file, period, genre))
        return xml_files
    
    def _process_single_file(self, xml_file: Path, period: str, genre: str) -> None:
        """Process a single XML file."""
        try:
            doc_data = self.process_gate_xml(xml_file, period, genre)
            if doc_data and doc_data['tokens']:
                self.documents.append(doc_data)
                self.stats['processed'] += 1
                self.logger.info(f"✓ Processed: {xml_file.name} ({len(doc_data['tokens'])} tokens)")
            else:
                self.stats['empty_files'] += 1
                self.logger.warning(f"No tokens found in: {xml_file.name}")
                
        except Exception as e:
            if "ParseError" in str(type(e)):
                self.stats['xml_errors'] += 1
            else:
                self.stats['processing_errors'] += 1
            self.logger.error(f"Error processing {xml_file.name}: {e}")
    
    def process_gate_xml(self, xml_path: Path, period: str, genre: str) -> Dict[str, Any]:
        """Process GATE XML file with TextWithNodes structure."""
        root = self.xml_parser.parse_xml_file(xml_path)
        if root is None:
            return None
        
        # Extract document metadata
        doc_id = xml_path.stem
        doc_data = {
            'doc_id': doc_id,
            'period': period,
            'genre': genre,
            'filename': xml_path.name,
            'year': self.text_processor.extract_year(xml_path.name),
            'region': self.text_processor.extract_region(xml_path.name),
            'text': '',
            'tokens': [],
            'sentences': [],
            'token_count': 0,
            'unique_words': set()
        }
        
        # Extract text and annotations
        text_data = self.xml_parser.extract_text_with_nodes(root)
        if text_data is None:
            return None
        
        text_content, node_positions = text_data
        doc_data['text'] = text_content
        
        annotations = self.xml_parser.extract_annotations(root)
        
        # Process tokens and sentences
        tokens, sentences = self.token_processor.extract_tokens_and_sentences(
            text_content, node_positions, annotations, doc_id, period, genre
        )
        
        doc_data['tokens'] = tokens
        doc_data['sentences'] = sentences
        doc_data['token_count'] = len(tokens)
        doc_data['unique_words'] = len(set(token['normalized'].lower() for token in tokens if token['normalized']))
        
        # Track linguistic changes
        for token in tokens:
            if token['is_spelling_variant']:
                self.linguistic_features['spelling_variants'].append({
                    'original': token['original'],
                    'normalized': token['normalized'],
                    'lemma': token['lemma'],
                    'period': period,
                    'genre': genre,
                    'pos': token['pos'],
                    'has_archaic_spelling': token['has_archaic_spelling']
                })
        
        # Add tokens to global list
        self.tokens.extend(tokens)
        
        return doc_data
