# ==========================================
# FILE: gate_preprocessor/token_processor.py
# ==========================================
"""Token processing and feature extraction."""

from typing import Dict, Any, Optional, List
from .linguistic_analyzer import LinguisticAnalyzer


class TokenProcessor:
    """Processes tokens and extracts linguistic features."""
    
    def __init__(self):
        self.analyzer = LinguisticAnalyzer()
    
    def create_token_data(self, token_text: str, features: Dict, doc_id: str, 
                         sent_id: int, token_id: int, period: str, genre: str) -> Optional[Dict[str, Any]]:
        """Create token data structure from text and features."""
        
        # Get linguistic features
        original = features.get('string', token_text)
        normalized = features.get('norm', original)
        lemma = features.get('lemma', normalized)
        pos = features.get('pos', '')
        morph = features.get('morph', '')
        
        # Skip empty tokens
        if len(original.strip()) == 0:
            return None
        
        token_data = {
            'doc_id': doc_id,
            'sentence_id': sent_id,
            'token_id': token_id,
            'period': period,
            'genre': genre,
            'original': original.strip(),
            'normalized': normalized.strip() if normalized else original.strip(),
            'lemma': lemma.strip() if lemma else normalized.strip() if normalized else original.strip(),
            'pos': pos,
            'morphology': morph,
            'is_spelling_variant': original.lower() != normalized.lower() if normalized else False,
            'word_length': len(original.strip()),
            'has_archaic_spelling': self.analyzer.is_archaic_spelling(original),
            'is_punctuation': self.analyzer.is_punctuation(original)
        }
        
        return token_data
    
    def extract_tokens_and_sentences(self, text: str, node_positions: Dict, 
                                   annotations: Dict, doc_id: str, period: str, genre: str) -> tuple:
        """Extract tokens and sentences from text and annotations."""
        from .annotation_extractor import AnnotationExtractor
        from .text_processor import TextProcessor
        
        extractor = AnnotationExtractor()
        text_proc = TextProcessor()
        
        tokens = []
        sentences = []
        
        token_annotations = extractor.extract_token_annotations(annotations)
        
        current_sentence = []
        sentence_id = 0
        token_id = 0
        
        for start_node, end_node, features, ann_id in token_annotations:
            token_text = extractor.get_text_span(text, node_positions, start_node, end_node)
            
            if not token_text:
                continue
            
            token_data = self.create_token_data(
                token_text, features, doc_id, sentence_id, token_id, period, genre
            )
            
            if token_data:
                tokens.append(token_data)
                current_sentence.append(token_data)
                token_id += 1
                
                # Check for sentence boundaries
                if text_proc.is_sentence_end(token_text):
                    if current_sentence:
                        sentence_text = ' '.join(t['original'] for t in current_sentence)
                        sentences.append({
                            'sentence_id': sentence_id,
                            'text': sentence_text,
                            'tokens': current_sentence.copy(),
                            'token_count': len(current_sentence)
                        })
                        current_sentence = []
                        sentence_id += 1
        
        # Handle remaining tokens as final sentence
        if current_sentence:
            sentence_text = ' '.join(t['original'] for t in current_sentence)
            sentences.append({
                'sentence_id': sentence_id,
                'text': sentence_text,
                'tokens': current_sentence,
                'token_count': len(current_sentence)
            })
        
        return tokens, sentences
