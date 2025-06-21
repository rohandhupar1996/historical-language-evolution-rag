# ==========================================
# FILE: gate_preprocessor/config.py
# ==========================================
"""Configuration for GATE XML preprocessor."""

from typing import List

# Default directories
DEFAULT_CONFIG = {
    'organized_dir': './data/organized',
    'output_dir': './data/processed',
    'encoding': 'utf-8',
    'file_extension': '.xml'
}

# Archaic spelling patterns for Early New High German
ARCHAIC_PATTERNS: List[str] = [
    r'.*uo.*',      # uo diphthong (guot -> gut)
    r'.*ie.*',      # ie for long i (liebe)
    r'.*ey.*',      # ey/ei variations
    r'.*ck$',       # archaic endings
    r'^v[aeiou]',   # v- beginnings (vmb -> um)
    r'.*th.*',      # th spellings (thun -> tun)
    r'.*umb$',      # umb endings (vmb -> um)
    r'.*tz$',       # tz endings
    r'.*ff.*',      # double f
    r'.*ss.*',      # double s patterns
    r'.*ů.*',       # archaic u with circle
    r'.*ä.*',       # archaic a-umlaut forms
    r'.*ö.*',       # archaic o-umlaut forms
    r'.*ü.*',       # archaic u-umlaut forms
]

# Sentence boundary markers
SENTENCE_ENDERS = '.!?;'

# Punctuation characters
PUNCTUATION_CHARS = '.,!?;:()[]{}"\'-'

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'