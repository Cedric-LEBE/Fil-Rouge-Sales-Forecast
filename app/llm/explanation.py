from __future__ import annotations

import os
import re
from typing import Any, Optional, Literal

import pandas as pd


Metric = Literal["sales", "clients", "unknown"]
Grain = Optional[Literal["day", "week", "month", "quarter", "year"]]


# ----------------------------
# Small inference helpers
# ----------------------------
def _infer_grain_from_sql(sql: str) -> Grain:
    s = (sql or "").lower()
    m = re.search(r"date_trunc\(\s*'(\w+)'\s*,\s*date\)", s)
    if not m:
        return None
    g = m.group(1)
    if g in {"day", "week", "month", "quarter", "year"}:
        return g  
    return None


def _infer_metric_from_df_sql(df: pd.DataFrame, sql: str) -> Metric:
    cols = {c.lower() for c in df.columns}
    s = (sql or "").lower()

    if "count(distinct customer_id)" in s or "daily_clients" in s or any("client" in c for c in cols):
        return "clients"
    if "sales" in s or any("sales" in c for c in cols):
        return "sales"
    return "unknown"


def _is_avg_per_period(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return "avg_value_per_period" in cols


def _is_timeseries(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return "period" in cols or "date" in cols


def _fmt_number(x: float) -> str:
    try:
        return f"{float(x):,.2f}".replace(",", " ").replace(".", ",")
    except Exception:
        return str(x)


def _fmt_money(x: float) -> str:
    try:
        # $ with no decimals for big numbers, 2 decimals otherwise
        v = float(x)
        if abs(v) >= 1000:
            s = f"{v:,.0f}".replace(",", " ")
        else:
            s = f"{v:,.2f}".replace(",", " ").replace(".", ",")
        return f"${s}"
    except Exception:
        return f"${x}"


def _unit(metric: Metric) -> str:
    if metric == "sales":
        return "$"
    if metric == "clients":
        return "clients"
    return "valeur"


def _grain_label(grain: Grain) -> str:
    return {
        "day": "jour",
        "week": "semaine",
        "month": "mois",
        "quarter": "trimestre",
        "year": "année",
        None: "période",
    }[grain]


def _pick_key_value(df: pd.DataFrame) -> Optional[tuple[str, float]]:
    """
    Return (colname, value) for the first numeric col.
    Used for totals / single-row results.
    """
    if df is None or df.empty:
        return None
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                return c, float(df.iloc[0][c])
            except Exception:
                continue
    return None


# ----------------------------
# Main
# ----------------------------
def explain_result(
    *,
    question: str,
    sql: str,
    df: pd.DataFrame,
    chart_hint: str | None = None,
) -> str:
    """LLM explanation layer with strong deterministic "analytics semantics".

    - Detects metric (sales vs clients)
    - Detects grain (day/week/month/quarter/year) from SQL date_trunc
    - Formats sales as $ and clients as "clients"
    - Adds a compact business summary even without LLM
    """

    if df is None or df.empty:
        return "Je n'ai trouvé aucun résultat pour cette requête."

    grain = _infer_grain_from_sql(sql)
    metric = _infer_metric_from_df_sql(df, sql)
    unit = _unit(metric)

    # ----------------------------
    # Deterministic summary (always)
    # ----------------------------
    lines: list[str] = []

    # Case A: AVG per period
    if _is_avg_per_period(df):
        # can be global (1 row) or grouped (multiple rows)
        if len(df) == 1:
            v = float(df.iloc[0]["avg_value_per_period"])
            v_fmt = _fmt_money(v) if metric == "sales" else _fmt_number(v)
            lines.append(
                f"- Moyenne **par {_grain_label(grain)}** : **{v_fmt}** ({unit}/{_grain_label(grain)})"
                if metric == "sales"
                else f"- Moyenne **par {_grain_label(grain)}** : **{v_fmt}** ({unit}/{_grain_label(grain)})"
            )
        else:
            # show top 5 groups by avg
            cols_lower = {c.lower(): c for c in df.columns}
            group_col = cols_lower.get("region") or cols_lower.get("macro_category") or None
            df_sorted = df.sort_values("avg_value_per_period", ascending=False)
            top = df_sorted.head(5)
            bullets = []
            for _, r in top.iterrows():
                name = str(r[group_col]) if group_col else "groupe"
                v = float(r["avg_value_per_period"])
                v_fmt = _fmt_money(v) if metric == "sales" else _fmt_number(v)
                bullets.append(f"  - {name}: {v_fmt}")
            lines.append(
                f"- Moyenne **par {_grain_label(grain)}** (Top 5 {group_col or 'groupes'}) :\n" + "\n".join(bullets)
            )

    # Case B: timeseries
    elif _is_timeseries(df) and len(df) > 1:
        # Identify value column
        val_col = None
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in {"year", "month"}:
                val_col = c
                break
        if val_col:
            total = float(df[val_col].sum())
            avg = float(df[val_col].mean())
            total_fmt = _fmt_money(total) if metric == "sales" else _fmt_number(total)
            avg_fmt = _fmt_money(avg) if metric == "sales" else _fmt_number(avg)
            lines.append(f"- Série temporelle détectée (grain: **{_grain_label(grain)}**).")
            lines.append(f"- Total sur la période : **{total_fmt}**")
            lines.append(f"- Moyenne par point de temps : **{avg_fmt}** ({unit}/{_grain_label(grain)})")

    # Case C: single KPI (sum, count, etc.)
    else:
        kv = _pick_key_value(df)
        if kv:
            k, v = kv
            v_fmt = _fmt_money(v) if metric == "sales" else _fmt_number(v)
            if metric == "sales":
                lines.append(f"- {k} : **{v_fmt}**")
            elif metric == "clients":
                lines.append(f"- {k} : **{v_fmt} {unit}**")
            else:
                lines.append(f"- {k} : **{v_fmt}**")

    # add context line
    ctx = []
    if grain:
        ctx.append(f"grain={grain}")
    if metric != "unknown":
        ctx.append(f"metric={metric}")

    deterministic = "\n".join(lines)

    # ----------------------------
    # If no GROQ, return deterministic only
    # ----------------------------
    if not os.getenv("GROQ_API_KEY"):
        return deterministic

    # ----------------------------
    # LLM enrichment
    # ----------------------------
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    sample = df.head(30).to_dict(orient="records")

    system = (
        "Tu es un analyste data senior. Réponds en français.\n"
        "- Réponds en TEXTE BRUT uniquement (pas de Markdown, pas de HTML).\n"
        "Ta mission: expliquer les résultats de manière claire, concise et orientée business.\n"
        "RÈGLES IMPORTANTES:\n"
        "- Ne pas inventer des faits au-delà du résultat fourni.\n"
        "- Ne répète pas la question.\n"
        "- Ne montre jamais des champs techniques (n_rows, columns, total_rows, etc.).\n"
        "- N'invente rien: base-toi uniquement sur result_preview + deterministic_summary.\n"
        "- Utilise le contexte détecté (metric/grain/unit) pour parler correctement:\n"
        "- Si metric='sales', parle simplement de ventes.\n"
        "  * metric='sales' => monnaie en dollars ($)\n"
        "  * metric='clients' => parler en 'clients'\n"
        "- Si l'utilisateur demande 'pourquoi', propose 2-3 hypothèses plausibles, formulées comme hypothèses.\n"
    )

    user_payload: dict[str, Any] = {
        "question": question,
        "sql": sql,
        "chart_hint": chart_hint,
        "detected": {"metric": metric, "unit": unit, "grain": grain},
        "result_preview": sample,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "deterministic_summary": deterministic,
    }

    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user_payload)},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    llm = (resp.choices[0].message.content or "").strip()


    if os.getenv("GROQ_API_KEY"):
        return llm  
    return deterministic