# ==========================================
# FILE: validation_suite/validator.py
# ==========================================
"""Main validation class."""

from pathlib import Path
from typing import Dict, Any
from .data_loader import DataLoader
from .temporal_validator import TemporalValidator
from .spelling_validator import SpellingValidator
from .linguistic_validator import LinguisticValidator
from .quality_validator import QualityValidator
from .rag_readiness_checker import RAGReadinessChecker
from .query_tester import QueryTester
from .report_generator import ReportGenerator
from .utils import setup_logging, print_validation_summary
from .config import ESSENTIAL_FIELDS


class GerManCValidator:
    """Main validation class for GerManC preprocessing."""
    
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.validation_results = {}
        self.critical_errors = []
        self.warnings = []
        
        # Initialize components
        self.data_loader = DataLoader(self.processed_dir)
        self.temporal_validator = TemporalValidator()
        self.spelling_validator = SpellingValidator()
        self.linguistic_validator = LinguisticValidator()
        self.quality_validator = QualityValidator()
        self.rag_checker = RAGReadinessChecker()
        self.query_tester = QueryTester()
        self.report_generator = ReportGenerator(self.processed_dir)
        
        setup_logging()
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🔍 Running GerManC Preprocessing Validation Suite...")
        print("=" * 60)
        
        # Load data
        self.documents, self.tokens_df, self.linguistic_features, self.statistics = self.data_loader.load_all_data()
        
        # Run validations
        self._validate_temporal_data()
        self._validate_spelling_variants()
        self._validate_linguistic_features()
        self._validate_text_quality()
        self._validate_data_completeness()
        self._validate_rag_readiness()
        self._test_sample_queries()
        
        # Generate report
        report = self.report_generator.generate_report(
            self.validation_results, self.critical_errors, self.warnings
        )
        
        print_validation_summary(self.critical_errors, self.warnings, self.validation_results)
        
        return self.validation_results
    
    def _validate_temporal_data(self):
        """Validate temporal distribution."""
        print("\n📅 Validating Temporal Data...")
        result = self.temporal_validator.validate(self.documents)
        self.validation_results['temporal'] = result
        
        if not result['valid']:
            self.critical_errors.append("Insufficient temporal periods for evolution tracking")
        
        print(f"   ✓ Periods: {result['periods_found']}")
        print(f"   ✓ Year range: {result['year_range']}")
    
    def _validate_spelling_variants(self):
        """Validate spelling variants."""
        print("\n🔤 Validating Spelling Variants...")
        result = self.spelling_validator.validate(self.linguistic_features, self.tokens_df)
        self.validation_results['spelling_variants'] = result
        
        if 'critical_error' in result:
            self.critical_errors.append(result['critical_error'])
        elif not result['valid']:
            self.warnings.append("Low spelling variant rate for historical texts")
        
        print(f"   ✓ Total variants: {result['total_variants']}")
        print(f"   ✓ Variant rate: {result['variant_rate']:.2%}")
    
    def _validate_linguistic_features(self):
        """Validate linguistic features."""
        print("\n🏷️ Validating Linguistic Features...")
        result = self.linguistic_validator.validate(self.tokens_df)
        self.validation_results['linguistic_features'] = result
        
        if not result['valid']:
            self.warnings.append("Few major POS categories found")
        
        print(f"   ✓ POS tags: {result['pos_tag_count']}")
        print(f"   ✓ Major POS: {result['major_pos_found']}/7")
    
    def _validate_text_quality(self):
        """Validate text quality."""
        print("\n📝 Validating Text Quality...")
        result = self.quality_validator.validate(self.documents, self.tokens_df)
        self.validation_results['text_quality'] = result
        
        print(f"   ✓ Empty docs: {result['empty_documents']}")
        print(f"   ✓ Avg token length: {result['average_token_length']:.1f}")
    
    def _validate_data_completeness(self):
        """Validate data completeness."""
        print("\n📊 Validating Data Completeness...")
        
        missing_fields = []
        if not self.tokens_df.empty:
            for field in ESSENTIAL_FIELDS:
                if field not in self.tokens_df.columns:
                    missing_fields.append(field)
        
        if missing_fields:
            self.critical_errors.append(f"Missing essential fields: {missing_fields}")
        
        genres = self.tokens_df['genre'].value_counts() if 'genre' in self.tokens_df.columns else {}
        
        self.validation_results['completeness'] = {
            'total_tokens': len(self.tokens_df),
            'missing_fields': missing_fields,
            'genre_distribution': dict(genres)
        }
        
        print(f"   ✓ Total tokens: {len(self.tokens_df)}")
        print(f"   ✓ Missing fields: {missing_fields}")
    
    def _validate_rag_readiness(self):
        """Validate RAG readiness."""
        print("\n🤖 Validating RAG Readiness...")
        result = self.rag_checker.check_readiness(self.validation_results, self.tokens_df, self.documents)
        self.validation_results['rag_readiness'] = result
        
        print(f"   📊 Readiness Score: {result['readiness_score']}/100")
        print(f"   {'✅ READY' if result['is_ready'] else '❌ NOT READY'}")
    
    def _test_sample_queries(self):
        """Test sample queries."""
        print("\n🔍 Testing Sample RAG Queries...")
        result = self.query_tester.test_queries(
            self.documents, self.tokens_df, self.linguistic_features, self.validation_results
        )
        self.validation_results['sample_queries'] = result
        
        for query, data in result.items():
            status = "✅" if data['can_answer'] else "❌"
            print(f"   {status} '{query[:40]}...' - {data['evidence_count']} examples")