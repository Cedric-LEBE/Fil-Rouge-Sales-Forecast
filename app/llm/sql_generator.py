from __future__ import annotations

import os
import re
from typing import Optional, Literal


TimeGrain = Literal["day", "week", "month", "quarter", "year"]


# ----------------------------
# Detection helpers
# ----------------------------
def _looks_like_forecast(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["prévision", "prevision", "forecast", "prédire", "predire", "prediction"])


def _extract_top_n(q: str, default: int = 5, max_n: int = 50) -> int:
    m = re.search(r"\btop\s*(\d+)\b", q.lower())
    if not m:
        return default
    n = int(m.group(1))
    return max(1, min(n, max_n))


def _extract_date_range(q: str) -> tuple[Optional[str], Optional[str]]:
    dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if len(dates) >= 2:
        return dates[0], dates[1]
    return None, None


def _detect_time_grain(q: str, default: TimeGrain = "day") -> TimeGrain:
    ql = q.lower()
    # more specific first
    if any(k in ql for k in ["par année", "par an", "annuel", "annuelle", "year", "yearly", "annuellement"]):
        return "year"
    if any(k in ql for k in ["trimestre", "quarter", "quarterly", "par trimestre"]):
        return "quarter"
    if any(k in ql for k in ["par mois", "mensuel", "mensuelle", "month", "monthly"]):
        return "month"
    if any(k in ql for k in ["semaine", "hebdo", "hebdomadaire", "week", "weekly"]):
        return "week"
    if any(k in ql for k in ["par jour", "quotidien", "quotidienne", "day", "daily"]):
        return "day"
    return default


def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # FR aliases -> EN canonical tokens
    s = s.replace("sud est", "southeast").replace("sudest", "southeast").replace("sud-est", "southeast")
    s = s.replace("nord est", "northeast").replace("nord-est", "northeast")
    s = s.replace("centre ouest", "central west").replace("centre-ouest", "central west")
    s = s.replace("central west", "central west")
    return s


def _detect_region_from_known(q: str, available: Optional[list[str]] = None) -> Optional[str]:
    """
    Detect region robustly, avoiding south matching before southeast.
    If `available` is provided, uses it (recommended). Otherwise fallback to known list.
    """
    qn = _normalize_text(q)

    if available:
        mapping = {_normalize_text(r): r for r in available}
        keys = sorted(mapping.keys(), key=len, reverse=True)  # longest first
        for k in keys:
            if re.search(rf"\b{re.escape(k)}\b", qn):
                return mapping[k]
        return None

    # fallback known tokens (longest first)
    known = [
        "central west",
        "southeast",
        "northeast",
        "north",
        "south",
    ]
    for k in sorted(known, key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", qn):
            return k
    return None


def _is_clients_metric(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["client", "clients", "customer", "customers", "acheteur", "acheteurs"])


def _is_sales_metric(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["vente", "ventes", "sales", "chiffre", "ca", "revenue", "revenu"])


def _wants_average(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["moyenne", "average", "avg", "en moyenne"])


def _wants_total(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["total", "totale", "totales", "sum", "somme"])


def _wants_timeseries(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["au fil du temps", "time series", "timeline", "évolution", "evolution", "trend", "tendance"])


def _wants_top(q: str) -> bool:
    ql = q.lower()
    return "top" in ql or "meilleur" in ql or "meilleure" in ql or "max" in ql or "plus" in ql


def _group_dimension(q: str) -> Optional[Literal["region", "macro_category"]]:
    ql = q.lower()
    if any(k in ql for k in ["catégorie", "categorie", "category", "macro", "produit", "products"]):
        return "macro_category"
    if any(k in ql for k in ["région", "region", "régions", "regions"]):
        return "region"
    return None


# ----------------------------
# SQL building helpers
# ----------------------------
def _where_clause(region: Optional[str], d0: Optional[str], d1: Optional[str]) -> str:
    where = []
    if region:
        r = region.replace("%", "").replace("_", "")
        where.append(f"region ILIKE '%{r}%'")
    if d0 and d1:
        where.append(f"date BETWEEN '{d0}' AND '{d1}'")
    return ("WHERE " + " AND ".join(where) + "\n") if where else ""


def _bucket_expr(grain: TimeGrain) -> str:

    return f"date_trunc('{grain}', date)::date"


def _metric_expr(metric: Literal["sales", "clients"]) -> str:
    if metric == "sales":
        return "SUM(sales)"
    return "COUNT(DISTINCT customer_id)"


def _sanitize_sql(sql: str) -> str:
    s = (sql or "").strip()

    if "```" in s:
        s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE).strip()
        s = s.replace("```", "").strip()

    s = re.sub(r"^(sql\s*:|here\s+is\s+the\s+sql\s*:)\s*", "", s, flags=re.IGNORECASE).strip()

    parts = [p.strip() for p in s.split(";") if p.strip()]
    if not parts:
        return "UNSUPPORTED"
    s = parts[0] + ";"

    upper = s.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "UNSUPPORTED"

    banned = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE", "VACUUM"]
    if any(b in upper for b in banned):
        return "UNSUPPORTED"

    return s


# ----------------------------
# Heuristic SQL (fallback)
# ----------------------------
def _heuristic_sql(question: str, *, schema_hint: str) -> Optional[str]:
    q = question.lower()
    top_n = _extract_top_n(question, default=5)
    d0, d1 = _extract_date_range(question)
    grain = _detect_time_grain(question, default="day")

    # metric
    if _is_clients_metric(question) and not _is_sales_metric(question):
        metric: Literal["sales", "clients"] = "clients"
    else:
        metric = "sales"

    # region detection (fallback list)
    region = _detect_region_from_known(question)

    where_sql = _where_clause(region, d0, d1)
    group_dim = _group_dimension(question)

    # Moyenne des clients "par jour" 
    if "moyenne" in q and ("par jour" in q or "quotid" in q or "daily" in q) and ("client" in q or "customers" in q):
        return (
            "WITH daily AS (\n"
            "  SELECT date, COUNT(DISTINCT customer_id) AS daily_clients\n"
            "  FROM sales\n"
            f"  {where_sql}"
            "  GROUP BY date\n"
            ")\n"
            "SELECT AVG(daily_clients) AS avg_value_per_period\n"
            "FROM daily;"
        )

    # -------------------------
    # 1) AVERAGE per period 
    # EX. "moyenne des ventes par jour/mois/année"
    # -------------------------
    if _wants_average(question):
        bucket = _bucket_expr(grain)
        inner_metric = _metric_expr(metric)

        if group_dim in ("region", "macro_category"):
            dim = group_dim
            return (
                "SELECT {dim}, AVG(bucket_value) AS avg_value_per_period\n"
                "FROM (\n"
                "  SELECT {bucket} AS bucket, {dim}, {inner_metric} AS bucket_value\n"
                "  FROM sales\n"
                "  {where}"
                "  GROUP BY 1, 2\n"
                ") t\n"
                "GROUP BY {dim}\n"
                "ORDER BY avg_value_per_period DESC;"
            ).format(dim=dim, bucket=bucket, inner_metric=inner_metric, where=where_sql)

        return (
            "SELECT AVG(bucket_value) AS avg_value_per_period\n"
            "FROM (\n"
            "  SELECT {bucket} AS bucket, {inner_metric} AS bucket_value\n"
            "  FROM sales\n"
            "  {where}"
            "  GROUP BY 1\n"
            ") t;"
        ).format(bucket=bucket, inner_metric=inner_metric, where=where_sql)

    # -------------------------
    # 2) TOTAL (sales sum or clients distinct) global
    # -------------------------
    if _wants_total(question) and (metric in ("sales", "clients")) and not _wants_timeseries(question):
        if metric == "sales":
            return "SELECT SUM(sales) AS total_sales\nFROM sales\n{where}".format(where=where_sql).rstrip() + ";"
        return "SELECT COUNT(DISTINCT customer_id) AS total_clients\nFROM sales\n{where}".format(where=where_sql).rstrip() + ";"

    # -------------------------
    # 3) TIMESERIES (bucketed)
    # -------------------------
    if _wants_timeseries(question):
        bucket = _bucket_expr(grain)
        inner_metric = _metric_expr(metric)
        alias = "total_sales" if metric == "sales" else "total_clients"

        # by dim + time
        if group_dim in ("region", "macro_category"):
            dim = group_dim
            return (
                "SELECT {bucket} AS period, {dim}, {inner_metric} AS {alias}\n"
                "FROM sales\n"
                "{where}"
                "GROUP BY 1, 2\n"
                "ORDER BY period ASC, {dim} ASC;"
            ).format(bucket=bucket, dim=dim, inner_metric=inner_metric, alias=alias, where=where_sql)

        # global time series
        return (
            "SELECT {bucket} AS period, {inner_metric} AS {alias}\n"
            "FROM sales\n"
            "{where}"
            "GROUP BY 1\n"
            "ORDER BY period ASC;"
        ).format(bucket=bucket, inner_metric=inner_metric, alias=alias, where=where_sql)

    # -------------------------
    # 4) TOP N by dim (region/category)
    # -------------------------
    if _wants_top(question) and group_dim in ("region", "macro_category"):
        dim = group_dim
        if metric == "sales":
            val = "SUM(sales)"
            alias = "total_sales"
        else:
            val = "COUNT(DISTINCT customer_id)"
            alias = "total_clients"

        return (
            "SELECT {dim}, {val} AS {alias}\n"
            "FROM sales\n"
            "{where}"
            "GROUP BY {dim}\n"
            "ORDER BY {alias} DESC\n"
            "LIMIT {top_n};"
        ).format(dim=dim, val=val, alias=alias, where=where_sql, top_n=top_n)

    # -------------------------
    # 5) Fallback: breakdown by region (sales)
    # -------------------------
    if group_dim == "region" and metric == "sales":
        return (
            "SELECT region, SUM(sales) AS total_sales\n"
            "FROM sales\n"
            "{where}"
            "GROUP BY region\n"
            "ORDER BY total_sales DESC;"
        ).format(where=where_sql)

    return None


# ----------------------------
# Main public function
# ----------------------------
def generate_sql(question: str, *, schema_hint: str) -> str:
    """
    Return a single read-only SQL query for Postgres (SELECT/WITH only),
    or exactly 'UNSUPPORTED'.

    Uses Groq if GROQ_API_KEY is set, otherwise falls back to heuristics.
    """

    # If the user asks forecast, let the router handle it
    if _looks_like_forecast(question):
        return "UNSUPPORTED"

    # Heuristic fallback
    if not os.getenv("GROQ_API_KEY"):
        sql = _heuristic_sql(question, schema_hint=schema_hint)
        return sql if sql else "UNSUPPORTED"

    # Groq LLM
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    system = (
        "You are a senior analytics engineer. "
        "You translate user questions into a SINGLE read-only SQL query for Postgres.\n"
        "STRICT RULES:\n"
        "- Output ONLY SQL (no markdown, no explanation)\n"
        "- Only SELECT or WITH queries\n"
        "- Must query ONLY the tables/columns provided in the schema\n"
        "- No INSERT/UPDATE/DELETE/DDL\n"
        "- If the user asks for an AVERAGE per time period (day/week/month/quarter/year), you MUST compute it as:\n"
        "  AVG( bucket_total ) where bucket_total is computed by grouping by date_trunc(<grain>, date).\n"
        "- If the user asks 'average sales per day' do NOT use AVG(sales) over all rows.\n"
        "- If the user asks about clients, use COUNT(DISTINCT customer_id).\n"
        "- Use date_trunc('day'|'week'|'month'|'quarter'|'year', date)::date for time buckets.\n"
        "- If impossible, output exactly: UNSUPPORTED\n"
    )

    user = f"""Schema (Postgres):
{schema_hint}

User question: {question}

Write the SQL now.
"""

    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=450,
    )

    raw = (resp.choices[0].message.content or "").strip()
    sql = _sanitize_sql(raw)
    return sql