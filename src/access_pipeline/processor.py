# ==========================================
# FILE: access_pipeline/processor.py
# ==========================================
"""Main ACCESS processor class."""

from pathlib import Path
from typing import Dict
from .database_setup import DatabaseSetup
from .api_server import GerManCAPI
from .utils import setup_logging, print_database_ready, print_api_endpoints


class AccessPhaseSetup:
    """Main processor for ACCESS phase."""
    
    def __init__(self, prepare_output_dir: str, db_config: Dict):
        self.prepare_dir = Path(prepare_output_dir)
        self.db_config = db_config
        self.db_setup = DatabaseSetup(db_config, self.prepare_dir)
        
        setup_logging()
    
    def setup_database(self):
        """Complete database setup."""
        print("🗃️ Phase 4: ACCESS - Database Setup")
        print("=" * 40)
        
        self.db_setup.create_schema()
        self.db_setup.import_data()
        self.db_setup.create_indexes()
        
        print_database_ready(self.db_config)
    
    def start_api_server(self):
        """Start the API server."""
        print("\n🚀 Starting API server...")
        
        api = GerManCAPI(self.db_config)
        print_api_endpoints()
        api.run()