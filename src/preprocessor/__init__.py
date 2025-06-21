# ==========================================
# FILE: gate_preprocessor/__init__.py
# ==========================================
"""GATE XML Preprocessor Package."""

from .preprocessor import GateXMLPreprocessor
from .xml_parser import XMLParser
from .text_processor import TextProcessor
from .annotation_extractor import AnnotationExtractor
from .token_processor import TokenProcessor
from .linguistic_analyzer import LinguisticAnalyzer
from .data_saver import DataSaver
from .statistics_generator import StatisticsGenerator
from . import config

__version__ = "1.0.0"
__all__ = [
    "GateXMLPreprocessor",
    "XMLParser", 
    "TextProcessor",
    "AnnotationExtractor",
    "TokenProcessor",
    "LinguisticAnalyzer",
    "DataSaver",
    "StatisticsGenerator",
    "config"
]