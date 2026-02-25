from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def _default_db_url() -> str:
    # Used both locally and in docker-compose (analytics-db service)
    return "postgresql+psycopg2://analytics:analytics@analytics-db:5432/analytics"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = os.getenv("ANALYTICS_DATABASE_URL", _default_db_url())
    return create_engine(url, pool_pre_ping=True)
