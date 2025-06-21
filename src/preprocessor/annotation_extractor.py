# ==========================================
# FILE: gate_preprocessor/annotation_extractor.py
# ==========================================
"""Annotation extraction and processing."""

from typing import List, Tuple, Dict, Any


class AnnotationExtractor:
    """Extracts and processes annotations from GATE XML."""
    
    def extract_token_annotations(self, annotations: Dict) -> List[Tuple[int, int, Dict, str]]:
        """Extract token annotations sorted by position."""
        token_annotations = []
        
        for ann_id, ann in annotations.items():
            if ann['type'] == 'Token':
                token_annotations.append((
                    ann['start_node'], 
                    ann['end_node'], 
                    ann['features'], 
                    ann_id
                ))
        
        return sorted(token_annotations, key=lambda x: x[0])
    
    def get_text_span(self, text: str, node_positions: Dict[int, int], 
                      start_node: int, end_node: int) -> str:
        """Extract text span using node positions."""
        start_pos = node_positions.get(start_node, 0)
        end_pos = node_positions.get(end_node, len(text))
        
        if start_pos < len(text) and end_pos <= len(text) and start_pos < end_pos:
            return text[start_pos:end_pos].strip()
        
        return ""