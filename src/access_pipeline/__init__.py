# ==========================================
# FILE: access_pipeline/__init__.py
# ==========================================
"""ACCESS Pipeline Package - Phase 4 Database & API."""

from .processor import AccessPhaseSetup
from .api_server import GerManCAPI
from .database_setup import DatabaseSetup
from .query_handlers import QueryHandlers
from . import config

__version__ = "1.0.0"
__all__ = [
    "AccessPhaseSetup",
    "GerManCAPI",
    "DatabaseSetup",
    "QueryHandlers",
    "config"
]
