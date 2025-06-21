# ==========================================
# FILE: validation_suite/report_generator.py
# ==========================================
"""Validation report generation."""

import json
from pathlib import Path
from typing import Dict, Any, List


class ReportGenerator:
    """Generates validation reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
    
    def generate_report(self, validation_results: Dict, critical_errors: List[str], 
                       warnings: List[str]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'validation_summary': {
                'critical_errors': len(critical_errors),
                'warnings': len(warnings),
                'overall_status': 'READY' if not critical_errors else 'NEEDS_FIXES'
            },
            'critical_errors': critical_errors,
            'warnings': warnings,
            'detailed_results': validation_results,
            'recommendations': self._generate_recommendations(validation_results, critical_errors)
        }
        
        # Save report
        report_path = self.output_dir / "validation_report.json"
        report = self._convert_to_json_serializable(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(self, validation_results: Dict, critical_errors: List[str]) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        if critical_errors:
            recommendations.append("Fix all critical errors before building RAG system")
        
        if validation_results.get('spelling_variants', {}).get('total_variants', 0) < 100:
            recommendations.append("Improve spelling variant extraction")
        
        if len(validation_results.get('temporal', {}).get('periods_found', [])) < 3:
            recommendations.append("Add more temporal periods")
        
        if validation_results.get('rag_readiness', {}).get('readiness_score', 0) < 80:
            recommendations.append("Address RAG readiness issues")
        
        return recommendations
    
    def _convert_to_json_serializable(self, obj):
        """Convert pandas types to JSON serializable."""
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj