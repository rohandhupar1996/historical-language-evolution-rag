# ==========================================
# FILE: gate_preprocessor/xml_parser.py
# ==========================================
"""XML parsing utilities for GATE XML files."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging


class XMLParser:
    """Parser for GATE XML files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_xml_file(self, xml_path: Path) -> Optional[ET.Element]:
        """Parse XML file and return root element."""
        try:
            tree = ET.parse(xml_path)
            return tree.getroot()
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {xml_path.name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading {xml_path.name}: {e}")
            return None
    
    def extract_text_with_nodes(self, root: ET.Element) -> Optional[Tuple[str, Dict[int, int]]]:
        """Extract text content and node positions from TextWithNodes element."""
        text_with_nodes = root.find('.//TextWithNodes')
        if text_with_nodes is None:
            self.logger.warning("No TextWithNodes found")
            return None
        
        return self._parse_text_with_nodes(text_with_nodes)
    
    def extract_annotations(self, root: ET.Element) -> Dict:
        """Extract annotations from AnnotationSet element."""
        annotation_set = root.find('.//AnnotationSet')
        if annotation_set is None:
            return {}
        
        return self._parse_annotations(annotation_set)
    
    def _parse_text_with_nodes(self, text_with_nodes_elem: ET.Element) -> Tuple[str, Dict[int, int]]:
        """Parse TextWithNodes element to extract text and node positions."""
        text_parts = []
        node_positions = {}
        
        # Process all text and node elements
        for elem in text_with_nodes_elem:
            if elem.tag == 'Node':
                node_id = elem.get('id')
                if node_id:
                    current_pos = len(''.join(text_parts))
                    node_positions[int(node_id)] = current_pos
            elif elem.text:
                text_parts.append(elem.text)
            
            # Handle tail text after nodes
            if elem.tail:
                text_parts.append(elem.tail)
        
        # Handle text directly in TextWithNodes
        if text_with_nodes_elem.text:
            text_parts.insert(0, text_with_nodes_elem.text)
        
        full_text = ''.join(text_parts)
        return full_text, node_positions
    
    def _parse_annotations(self, annotation_set: ET.Element) -> Dict:
        """Parse annotation elements to extract linguistic features."""
        annotations = {}
        
        for annotation in annotation_set.findall('Annotation'):
            ann_id = annotation.get('Id')
            ann_type = annotation.get('Type')
            start_node = annotation.get('StartNode')
            end_node = annotation.get('EndNode')
            
            if not all([ann_id, ann_type, start_node, end_node]):
                continue
            
            # Extract features
            features = {}
            for feature in annotation.findall('Feature'):
                name_elem = feature.find('Name')
                value_elem = feature.find('Value')
                
                if name_elem is not None and value_elem is not None:
                    feature_name = name_elem.text
                    feature_value = value_elem.text
                    features[feature_name] = feature_value
            
            annotations[ann_id] = {
                'type': ann_type,
                'start_node': int(start_node),
                'end_node': int(end_node),
                'features': features
            }
        
        return annotations