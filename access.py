# ==========================================
# FILE: access.py
# ==========================================
"""Main entry point for ACCESS phase."""

import argparse
from pathlib import Path
from src.access_pipeline import AccessPhaseSetup
from src.access_pipeline.config import DEFAULT_DB_CONFIG


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Setup PostgreSQL database and API")
    parser.add_argument("prepare_dir", help="Directory with prepared data")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--db-name", default="germanc_corpus", help="Database name")
    parser.add_argument("--db-user", default="rohan", help="Database user")
    parser.add_argument("--db-password", default="1996", help="Database password")
    parser.add_argument("--start-api", action="store_true", help="Start API server after setup")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.prepare_dir).exists():
        print(f"Error: Directory {args.prepare_dir} does not exist")
        return 1
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Setup database
    setup = AccessPhaseSetup(args.prepare_dir, db_config)
    setup.setup_database()
    
    # Optionally start API server
    if args.start_api:
        setup.start_api_server()
    
    return 0


if __name__ == "__main__":
    exit(main())