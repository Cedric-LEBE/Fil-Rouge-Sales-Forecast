from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import sqlparse
from sqlalchemy import text

from .db import get_engine


class UnsafeSQLError(ValueError):
    pass


READONLY_KEYWORDS = {
    "select",
    "with",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "on",
    "having",
    "union",
}


def _is_readonly(sql: str) -> bool:
    s = sql.strip().rstrip(";")
    if not s:
        return False

    # Block multiple statements
    parsed = sqlparse.parse(s)
    if len(parsed) != 1:
        return False

    # Block obvious write/ddl words
    forbidden = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|call|do)\b",
        re.IGNORECASE,
    )
    if forbidden.search(s):
        return False

    # Must start with SELECT or WITH
    start = s.lstrip().lower()
    return start.startswith("select") or start.startswith("with")


@dataclass
class QueryResult:
    sql: str
    df: pd.DataFrame


def run_query(sql: str) -> QueryResult:
    if not _is_readonly(sql):
        raise UnsafeSQLError("Only a single read-only SELECT/WITH query is allowed.")

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return QueryResult(sql=sql, df=df)
