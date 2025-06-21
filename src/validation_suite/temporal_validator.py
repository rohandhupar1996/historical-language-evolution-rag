# ==========================================
# FILE: validation_suite/temporal_validator.py
# ==========================================
"""Temporal data validation."""

from typing import List, Dict, Any
from .config import THRESHOLDS


class TemporalValidator:
    """Validates temporal distribution and dating."""
    
    def validate(self, documents: List[Dict]) -> Dict[str, Any]:
        """Validate temporal data."""
        temporal_issues = []
        periods = set()
        years = []
        
        for doc in documents:
            if doc.get('period'):
                periods.add(doc['period'])
            if doc.get('year') and 1500 < doc['year'] < 2000:
                years.append(doc['year'])
            elif doc.get('year'):
                temporal_issues.append(f"Suspicious year: {doc['year']} in {doc['filename']}")
        
        year_range = max(years) - min(years) if years else 0
        
        return {
            'periods_found': list(periods),
            'year_range': year_range,
            'docs_with_years': len(years),
            'temporal_coverage': year_range > 200,
            'issues': temporal_issues,
            'valid': len(periods) >= 2 and year_range > 200
        }