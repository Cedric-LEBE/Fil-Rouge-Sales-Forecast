from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]

DATA_INTERIM = ROOT / "data" / "interim"
MODEL_LATEST = ROOT / "model_store" / "latest"

MERGED_PATH = DATA_INTERIM / "data_merged_clean.parquet"
REGION_BASE_PATH = DATA_INTERIM / "sales_region_day_base.parquet"
GLOBAL_BASE_PATH = DATA_INTERIM / "sales_global_day_base.parquet"

ML_GLOBAL_PATH = MODEL_LATEST / "ml_global" / "pipeline.joblib"
ML_REGION_PATH = MODEL_LATEST / "ml_region" / "pipeline.joblib"


# =========================
# Page config
# =========================
st.set_page_config(page_title="Retail Forecast", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      div[data-testid="metric-container"] {
        background: rgba(240,242,246,0.6);
        padding: 12px 12px;
        border-radius: 12px;
        border: 1px solid rgba(120,120,120,0.12);
      }
      .small-note { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 Retail Sales Dashboard & Forecast")


# =========================
# Caches
# =========================
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)


# =========================
# Forecast helper 
# =========================
def iterative_forecast_ml(
    pipeline,
    base_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_col: str | None = None,
    group_value: str | None = None,
    horizon: int = 30,
):
    base_df = base_df.copy()
    base_df[date_col] = pd.to_datetime(base_df[date_col], errors="coerce")
    base_df = base_df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if group_col and group_value is not None:
        base_df = base_df[base_df[group_col] == group_value].copy().reset_index(drop=True)

    if base_df.empty:
        raise ValueError("Historique vide après filtrage (vérifie la sélection).")

    last_date = base_df[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    ext = pd.DataFrame({date_col: pd.concat([base_df[date_col], pd.Series(future_dates)], ignore_index=True)})

    if group_col and group_value is not None:
        ext[group_col] = group_value

    ext[target_col] = pd.concat(
        [base_df[target_col].astype(float), pd.Series([np.nan] * horizon)],
        ignore_index=True,
    )

    # time features
    ext["Year"] = ext[date_col].dt.year
    ext["Month"] = ext[date_col].dt.month
    ext["Quarter"] = ext[date_col].dt.quarter
    ext["Day"] = ext[date_col].dt.day
    ext["DayOfWeek"] = ext[date_col].dt.dayofweek
    ext["DayOfYear"] = ext[date_col].dt.dayofyear
    ext["WeekOfYear"] = ext[date_col].dt.isocalendar().week.astype(int)
    ext["IsWeekend"] = ext["DayOfWeek"].isin([5, 6]).astype(int)

    lags = (1, 7, 14, 30)
    windows = (7, 14, 30)

    for i in range(len(base_df), len(ext)):
        for lag in lags:
            ext.loc[i, f"sales_lag_{lag}"] = ext.loc[i - lag, target_col] if i - lag >= 0 else np.nan

        for w in windows:
            vals = ext.loc[max(0, i - w): i - 1, target_col].dropna().astype(float)
            ext.loc[i, f"rolling_mean_{w}"] = vals.mean() if len(vals) >= 2 else np.nan
            ext.loc[i, f"rolling_std_{w}"] = vals.std(ddof=1) if len(vals) >= 2 else np.nan

        X_row = ext.loc[[i]].drop(columns=[target_col], errors="ignore")
        ext.loc[i, target_col] = float(pipeline.predict(X_row)[0])

    out = pd.DataFrame({date_col: ext[date_col]})
    out["y_true"] = np.nan
    out.loc[: len(base_df) - 1, "y_true"] = base_df[target_col].astype(float).values
    out["y_pred"] = np.nan
    out.loc[len(base_df):, "y_pred"] = ext.loc[len(base_df):, target_col].astype(float).values
    return out


# =========================
# Navigation in Sidebar
# =========================
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "📊 Dashboard"

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Aller vers",
        ["📊 Dashboard", "🔮 Prévision", "💬 Chatbot"],
        index=0,
        key="nav_page",
    )


# =========================
# Load merged once (for dashboard filters)
# =========================
try:
    merged = load_parquet(MERGED_PATH)
except Exception as e:
    st.error(
        f"Impossible de charger {MERGED_PATH.name}.\n\n"
        f"Lance d'abord: `python scripts/make_dataset.py`\n\n"
        f"Détail: {e}"
    )
    st.stop()

merged["Date"] = pd.to_datetime(merged.get("Date"), errors="coerce")
merged = merged.dropna(subset=["Date"])

min_d, max_d = merged["Date"].min(), merged["Date"].max()


# =========================
# Sidebar filters 
# =========================
d0, d1 = min_d.date(), max_d.date()
sel_regions: list[str] | None = None
sel_macros: list[str] | None = None

if page == "📊 Dashboard":
    with st.sidebar:
        st.header("Filtres Dashboard")

        d0, d1 = st.date_input("Période", value=(min_d.date(), max_d.date()))

        regions = sorted(merged["Region"].dropna().unique().tolist()) if "Region" in merged.columns else []
        sel_regions = st.multiselect("Régions", options=regions, default=regions)

        macros = sorted(merged["Macro_Category"].dropna().unique().tolist()) if "Macro_Category" in merged.columns else []
        sel_macros = st.multiselect("Catégories", options=macros, default=macros)


# =========================
# Dashboard data 
# =========================
df_dash = merged.copy()

if page == "📊 Dashboard":
    df_dash = merged[
        (merged["Date"] >= pd.to_datetime(d0)) & (merged["Date"] <= pd.to_datetime(d1))
    ].copy()

    if sel_regions is not None and "Region" in df_dash.columns:
        df_dash = df_dash[df_dash["Region"].isin(sel_regions)].copy()

    if sel_macros is not None and "Macro_Category" in df_dash.columns:
        df_dash = df_dash[df_dash["Macro_Category"].isin(sel_macros)].copy()


# =========================
# Page: Dashboard
# =========================
if page == "📊 Dashboard":

    tabs = st.tabs([
        "📊 Vue d’ensemble",
        "📈 Ventes au fil du temps",
        "🏪 Ventes par région",
        "🧺 Ventes par produit",
        "👥 Analyse clients",
        "💳 Paiements",
    ])


    # -------------------------------------------------
    # 1) Vue d’ensemble
    # -------------------------------------------------
    with tabs[0]:
        st.subheader("🌟 KPIs")

        total_sales = float(df_dash["Sales"].sum()) if "Sales" in df_dash.columns else 0.0
        daily_sales = (
            df_dash.groupby("Date")["Sales"].sum()
            if "Sales" in df_dash.columns and len(df_dash)
            else pd.Series(dtype=float)
        )

        total_clients = int(df_dash["customer_id"].nunique()) if "customer_id" in df_dash.columns else None
        daily_clients = df_dash.groupby("Date")["customer_id"].nunique() if "customer_id" in df_dash.columns else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{total_sales:,.0f}")
        c2.metric("Avg Sales / Day", f"{daily_sales.mean():,.0f}" if len(daily_sales) else "—")
        c3.metric("Total Clients", f"{total_clients:,.0f}" if total_clients is not None else "—")
        c4.metric("Avg Clients / Day", f"{daily_clients.mean():,.0f}" if daily_clients is not None and len(daily_clients) else "—")

        st.divider()

        colA, colB = st.columns([2, 1])

        with colA:
            st.subheader("Sales over time")
            if len(daily_sales):
                fig = px.line(
                    daily_sales.reset_index(name="Sales"),
                    x="Date",
                    y="Sales",
                    markers=False,
                )
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True, key="overview_sales_time")
            else:
                st.info("Pas de données pour cette période / filtres.")

        with colB:
            st.subheader("Top regions")
            if "Region" in df_dash.columns and "Sales" in df_dash.columns and len(df_dash):
                by_region = df_dash.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(by_region.head(10), x="Sales", y="Region", orientation="h")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True, key="overview_top_regions")
            else:
                st.info("Aucune info région disponible.")

        colC, colD = st.columns(2)

        with colC:
            st.subheader("Sales by Category")
            if "Macro_Category" in df_dash.columns and "Sales" in df_dash.columns and len(df_dash):
                by_cat = df_dash.groupby("Macro_Category")["Sales"].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(by_cat, x="Macro_Category", y="Sales")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, use_container_width=True, key="overview_cat")
            else:
                st.info("Aucune macro-catégorie disponible.")

        with colD:
            st.subheader("Sales by Payment Type")
            if "payment_type" in df_dash.columns and "Sales" in df_dash.columns and len(df_dash):

                by_pay = (
                    df_dash.groupby("payment_type")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                total = float(by_pay["Sales"].sum())
                if total <= 0:
                    st.info("Aucun montant de ventes exploitable pour afficher un donut.")
                else:
                    # % arrondi à 1 décimale + ajustement pour total=100.0
                    by_pay["pct"] = (by_pay["Sales"] / total * 100).round(1)
                    diff = round(100.0 - float(by_pay["pct"].sum()), 1)
                    by_pay.loc[by_pay.index[-1], "pct"] = round(float(by_pay.loc[by_pay.index[-1], "pct"]) + diff, 1)

                    fig = px.pie(
                        by_pay,
                        names="payment_type",
                        values="pct",
                        hole=0.45,
                    )
                    fig.update_traces(
                        texttemplate="%{value:.1f}%",
                        textposition="inside",
                        hovertemplate="%{label}=%{value:.1f}%<br>Sales=%{customdata[0]:,.0f}<extra></extra>",
                        customdata=by_pay[["Sales"]].to_numpy(),
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
                    st.plotly_chart(fig, use_container_width=True, key="overview_pay")
            else:
                st.info("Aucun type de paiement disponible.")

        st.subheader("Table (filtrée)")
        st.dataframe(df_dash, use_container_width=True, height=340)

    # -------------------------------------------------
    # 2) Ventes au fil du temps (daily/weekly/quarterly/monthly)
    # -------------------------------------------------
    with tabs[1]:
        st.subheader("📈 Ventes au fil du temps")

        if "Sales" not in df_dash.columns or "Date" not in df_dash.columns or df_dash.empty:
            st.info("Pas de données disponibles.")
        else:
            # ---------------- Daily ----------------
            daily = df_dash.groupby("Date")["Sales"].sum().reset_index()

            # ---------------- Weekly (ISO YYYY-WW) ----------------
            weekly = (
                df_dash.set_index("Date")["Sales"]
                .resample("W-MON")  # semaine ISO (lundi)
                .sum()
                .reset_index()
                .rename(columns={"Date": "WeekDay"})
            )

            iso = weekly["WeekDay"].dt.isocalendar()
            weekly["YearWeek"] = (
                iso.year.astype(str)
                + "-"
                + iso.week.astype(str).str.zfill(2)
            )

            # ---------------- Monthly ----------------
            monthly = (
                df_dash.assign(Month=df_dash["Date"].dt.to_period("M").astype(str))
                .groupby("Month")["Sales"].sum().reset_index()
            )

            # ---------------- Quarterly ----------------
            quarterly = (
                df_dash.assign(Quarter=df_dash["Date"].dt.to_period("Q").astype(str))
                .groupby("Quarter")["Sales"].sum().reset_index()
            )

            c1, c2 = st.columns(2)

            # ===== DAILY =====
            with c1:
                fig = px.line(daily, x="Date", y="Sales", title="Ventes quotidiennes")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, key="ts_daily")

            # ===== WEEKLY (AXE + YYYY-WW) =====
            with c2:
                fig = px.line(
                    weekly,
                    x="WeekDay",
                    y="Sales",
                    title="Ventes hebdomadaires"
                )

                # Limiter le nombre de labels affichés
                step = max(len(weekly) // 15, 1)
                tick_idx = list(range(0, len(weekly), step))

                fig.update_xaxes(
                    tickmode="array",
                    tickvals=weekly.loc[tick_idx, "WeekDay"],
                    ticktext=weekly.loc[tick_idx, "YearWeek"],
                    tickangle=45,
                )

                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, key="ts_weekly")

            c3, c4 = st.columns(2)

            # ===== MONTHLY =====
            with c3:
                fig = px.line(monthly, x="Month", y="Sales", title="Ventes mensuelles")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, key="ts_monthly")

            # ===== QUARTERLY =====
            with c4:
                fig = px.line(quarterly, x="Quarter", y="Sales", title="Ventes trimestrielles")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, key="ts_quarterly")

    # -------------------------------------------------
    # 3) Ventes par région (global + au fil du temps)
    # -------------------------------------------------
    with tabs[2]:
        st.subheader("🏪 Ventes par région")

        if "Region" not in df_dash.columns or "Sales" not in df_dash.columns or df_dash.empty:
            st.info("Aucune donnée région disponible.")
        else:
            by_region = df_dash.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(by_region, x="Sales", y="Region", orientation="h", title="Ventes globales par région")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True, key="reg_global")

            st.divider()
            region_choice = st.selectbox("Choisir une région", ["Toutes"] + by_region["Region"].tolist(), key="reg_choice")

        dft = df_dash.copy()

        if region_choice == "Toutes":
            # 1 courbe par région
            ts = (
                dft.groupby(["Date", "Region"])["Sales"]
                .sum()
                .reset_index()
                .sort_values(["Date", "Region"])
            )

            fig = px.line(
                ts,
                x="Date",
                y="Sales",
                color="Region",
                title="Ventes au fil du temps — Toutes les régions",
            )

        else:
            # 1 seule courbe (région sélectionnée)
            dft = dft[dft["Region"] == region_choice]
            ts = (
                dft.groupby("Date")["Sales"]
                .sum()
                .reset_index()
                .sort_values("Date")
            )

            fig = px.line(
                ts,
                x="Date",
                y="Sales",
                title=f"Ventes au fil du temps — {region_choice}",
            )

        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True, key="reg_time")

    # -------------------------------------------------
    # 4) Ventes par produit (Macro_Category)
    # -------------------------------------------------
    with tabs[3]:
        st.subheader("🧺 Ventes par catégorie de produits")

        if "Sales" not in df_dash.columns or df_dash.empty:
            st.info("Pas de données disponibles.")
        else:
            prod_col = "Macro_Category" if "Macro_Category" in df_dash.columns else None
            if prod_col is None:
                st.info("Aucune colonne produit trouvée (Macro_Category).")
            else:

                by_prod = (
                    df_dash.groupby(prod_col)["Sales"].sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                fig = px.bar(by_prod, x="Sales", y=prod_col, orientation="h", title=f"Ventes globales par catégorie")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True, key="prod_global")

                st.divider()

                prod_choice = st.selectbox("Catégorie", by_prod[prod_col].tolist(), key="prod_choice")

                if "Region" in df_dash.columns:
                    reg_list = sorted(df_dash["Region"].dropna().unique().tolist())
                    reg_choice = st.selectbox("Région", ["Toutes"] + reg_list, key="prod_reg_choice")
                else:
                    reg_choice = "Toutes"

                dft = df_dash[df_dash[prod_col] == prod_choice].copy()

                if "Region" in dft.columns and reg_choice == "Toutes":
                    # 1 courbe par région (pour la catégorie sélectionnée)
                    ts = (
                        dft.groupby(["Date", "Region"])["Sales"]
                        .sum()
                        .reset_index()
                        .sort_values(["Date", "Region"])
                    )

                    fig = px.line(
                        ts,
                        x="Date",
                        y="Sales",
                        color="Region",
                        title=f"Ventes au fil du temps — {prod_choice} — Toutes les régions",
                    )

                else:
                    # 1 seule courbe (région sélectionnée ou pas de colonne Region)
                    if reg_choice != "Toutes" and "Region" in dft.columns:
                        dft = dft[dft["Region"] == reg_choice]

                    ts = (
                        dft.groupby("Date")["Sales"]
                        .sum()
                        .reset_index()
                        .sort_values("Date")
                    )

                    fig = px.line(
                        ts,
                        x="Date",
                        y="Sales",
                        title=f"Ventes au fil du temps — {prod_choice} — {reg_choice}",
                    )

                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, key="prod_time")

    # -------------------------------------------------
    # 5) Analyse clients
    # -------------------------------------------------
    with tabs[4]:
        st.subheader("👥 Analyse clients")

        if "customer_id" not in df_dash.columns or df_dash.empty:
            st.info("Aucune info client (customer_id) disponible.")
        else:
            daily_clients = df_dash.groupby("Date")["customer_id"].nunique().reset_index(name="clients_uniques")
            fig = px.line(daily_clients, x="Date", y="clients_uniques", title="Clients uniques (global) — au fil du temps")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True, key="clients_global")

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                if "Region" in df_dash.columns:
                    by_region = df_dash.groupby("Region")["customer_id"].nunique().sort_values(ascending=False).reset_index(name="clients_uniques")
                    fig = px.bar(by_region, x="clients_uniques", y="Region", orientation="h", title="Clients uniques par région")
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True, key="clients_region")
                else:
                    st.info("Pas de colonne Region.")

            with c2:
                if "Macro_Category" in df_dash.columns:
                    by_prod = (
                        df_dash.groupby("Macro_Category")["customer_id"].nunique()
                        .sort_values(ascending=False).head(15)
                        .reset_index(name="clients_uniques")
                    )
                    fig = px.bar(by_prod, x="clients_uniques", y="Macro_Category", orientation="h", title="Clients uniques par catégories")
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True, key="clients_prod")
                else:
                    st.info("Pas de colonne Macro_Category détectée.")

    # -------------------------------------------------
    # 6) Paiements
    # -------------------------------------------------
    with tabs[5]:
        st.subheader("💳 Analyse des types de paiements")

        if "payment_type" not in df_dash.columns or "Sales" not in df_dash.columns or df_dash.empty:
            st.info("Aucune info paiement (payment_type) disponible.")
        else:
            by_pay = (
                df_dash.groupby("payment_type")["Sales"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            total = float(by_pay["Sales"].sum())
            if total <= 0:
                st.info("Aucun montant de ventes exploitable pour afficher un donut.")
            else:
                # % arrondi à 1 décimale + ajustement pour total=100.0
                by_pay["pct"] = (by_pay["Sales"] / total * 100).round(1)
                diff = round(100.0 - float(by_pay["pct"].sum()), 1)
                by_pay.loc[by_pay.index[-1], "pct"] = round(float(by_pay.loc[by_pay.index[-1], "pct"]) + diff, 1)

                fig = px.pie(
                    by_pay,
                    names="payment_type",
                    values="pct",
                    hole=0.45,
                    title="Ventes globales par type de paiement",
                )
                fig.update_traces(
                    texttemplate="%{value:.1f}%",
                    textposition="inside",
                    hovertemplate="%{label}=%{value:.1f}%<br>Sales=%{customdata[0]:,.0f}<extra></extra>",
                    customdata=by_pay[["Sales"]].to_numpy(),
                )
                fig.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=300)
                st.plotly_chart(fig, use_container_width=False, key="pay_global")

            st.divider()

            if "Macro_Category" not in df_dash.columns:
                st.info("Pas de colonne Macro_Category détectée pour l'analyse paiement par catégorie.")
            else:
                prod_choice = st.selectbox(
                    "Catégorie",
                    sorted(df_dash["Macro_Category"].dropna().unique().tolist())[:200],
                    key="pay_prod_choice",
                )
                dft = df_dash[df_dash["Macro_Category"] == prod_choice].copy()

                by_pay_prod = dft.groupby("payment_type")["Sales"].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(by_pay_prod, x="Sales", y="payment_type", orientation="h",
                             title=f"Type de paiement — {prod_choice}")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, key="pay_by_prod")


# =========================
# Page: Prediction
# =========================
elif page == "🔮 Prévision":

    st.subheader("🔮 Prévision des ventes")

    # Onglets Prévision
    tabs_pred = st.tabs([
        "🌍 Prévision globale — Vue exécutive",
        "📈 Prévision des ventes",
    ])

    # =================================================
    # 1) Prévision des ventes 
    # =================================================
    with tabs_pred[1]:
        level = st.selectbox("Niveau", ["Global", "Région"], key="pred_level")
        horizon = st.slider("Horizon (jours)", 7, 180, 30, 1, key="pred_horizon")

        run = st.button("Lancer la Prévision", type="primary", key="pred_run")

        if level == "Global":
            if not GLOBAL_BASE_PATH.exists():
                st.error("Missing base file: sales_global_day_base.parquet (lance make_dataset)")
                st.stop()
            if not ML_GLOBAL_PATH.exists():
                st.error("Modèle global introuvable. Lance: `python scripts/train_ml_global.py`")
                st.stop()

            base = load_parquet(GLOBAL_BASE_PATH)
            base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
            pipe = load_joblib(ML_GLOBAL_PATH)

            if run:
                base = base.copy()
                base["Region"] = "GLOBAL"

                out = iterative_forecast_ml(
                    pipe,
                    base_df=base,
                    date_col="Date",
                    target_col="daily_sales_global",
                    group_col="Region",
                    group_value="GLOBAL",
                    horizon=horizon,
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=out["Date"], y=out["y_true"], mode="lines", name="Historique"))
                fig.add_trace(go.Scatter(x=out["Date"], y=out["y_pred"], mode="lines", name="Prévision", line=dict(color="red")))
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True, key="pred_global_chart")

                st.subheader("Prévisions")

                df_display = (
                    out.tail(horizon)[["Date", "y_pred"]]
                    .rename(columns={
                        "y_pred": "Predicted_Sales"
                    })
                )
                df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(df_display, use_container_width=True)

        else:
            if not REGION_BASE_PATH.exists():
                st.error("Missing base file: sales_region_day_base.parquet (lance make_dataset)")
                st.stop()
            if not ML_REGION_PATH.exists():
                st.error("Modèle région introuvable. Lance: `python scripts/train_ml_region.py`")
                st.stop()

            base = load_parquet(REGION_BASE_PATH)
            base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
            regions = sorted(base["Region"].dropna().unique().tolist())
            region = st.selectbox("Région", regions, key="pred_region")

            pipe = load_joblib(ML_REGION_PATH)

            if run:
                out = iterative_forecast_ml(
                    pipe,
                    base_df=base,
                    date_col="Date",
                    target_col="daily_sales",
                    group_col="Region",
                    group_value=region,
                    horizon=horizon,
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=out["Date"], y=out["y_true"], mode="lines", name="Historique"))
                fig.add_trace(go.Scatter(x=out["Date"], y=out["y_pred"], mode="lines", name="Prévision", line=dict(color="red")))
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True, key="pred_region_chart")

                st.subheader("Prévisions")

                df_display = (
                    out.tail(horizon)[["Date", "y_pred"]]
                    .rename(columns={
                        "y_pred": "Predicted_Sales"
                    })
                )
                df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(df_display, use_container_width=True)

    # =================================================
    # 2) Prévision globale — Vue exécutive (KPIs + dashboard)
    # =================================================
    with tabs_pred[0]:
        st.subheader("🌍 Prévision globale — Vue exécutive")

        if not GLOBAL_BASE_PATH.exists():
            st.error("Missing base file: sales_global_day_base.parquet (lance make_dataset)")
            st.stop()
        if not ML_GLOBAL_PATH.exists():
            st.error("Modèle global introuvable. Lance: `python scripts/train_ml_global.py`")
            st.stop()

        base = load_parquet(GLOBAL_BASE_PATH)
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"]).sort_values("Date")

        pipe = load_joblib(ML_GLOBAL_PATH)

        horizon_months = st.slider("Horizon (mois)", 1, 12, 6, key="pred_exec_months")
        horizon_days = int(horizon_months * 30)

        base = base.copy()
        base["Region"] = "GLOBAL"

        out = iterative_forecast_ml(
            pipe,
            base_df=base,
            date_col="Date",
            target_col="daily_sales_global",
            group_col="Region",
            group_value="GLOBAL",
            horizon=horizon_days,
        )

        past = out.dropna(subset=["y_true"]).copy()
        future = out.dropna(subset=["y_pred"]).copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Ventes prévues (total)", f"{future['y_pred'].sum():,.0f}")
        c2.metric("📆 Moyenne journalière prévue", f"{future['y_pred'].mean():,.0f}")
        c3.metric("💵 Dernière vente observée", f"{past['y_true'].iloc[-1]:,.0f}")

        st.divider()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=past["Date"], y=past["y_true"], mode="lines", name="Historique"))
        fig.add_trace(go.Scatter(x=future["Date"], y=future["y_pred"], mode="lines", name="Prévision", line=dict(color="red")))
        fig.update_layout(
            title=f"Prévision globale des ventes — {horizon_months} mois",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True, key="pred_exec_chart")

        st.subheader("Détail des prévisions (dernier 90 jours)")

        df_display = (
            future.tail(90)[["Date", "y_pred"]]
            .rename(columns={
                "Date": "Date",
                "y_pred": "Predicted_Sales"
            })
        )
        df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(df_display, use_container_width=True)

# =========================
# Page: Chatbot — Analytics SQL + Forecast ML
# =========================
elif page == "💬 Chatbot":
    st.subheader("💬 Chatbot — Analytics & Forecast")

    import re
    import plotly.express as px
    import plotly.graph_objects as go

    from analytics.query_executor import UnsafeSQLError, run_query
    from llm.sql_generator import generate_sql
    from llm.explanation import explain_result

    # Forecast engine (joblib + parquet)
    from forecast.ml_forecast import forecast_sales

    SCHEMA_HINT = """
Table: sales
- date (date)
- region (text)
- macro_category (text)
- sales (numeric)
- payment_type (text, optional)
- customer_id (text/int, optional)
""".strip()

    # -------------------------
    # Sidebar settings
    # -------------------------
    with st.sidebar:
        st.caption("⚙️ Chatbot settings")
        show_sql = st.toggle("Afficher le SQL", value=False)
        #max_rows = st.slider("Max lignes affichées", 50, 2000, 300, 50)
        default_horizon = st.slider("Horizon forecast par défaut (jours)", 7, 365, 30, 1)


    # --- Aperçu rapide des données disponibles ---
    with st.expander("📦 Voir un aperçu du dataset", expanded=False):
        st.write(f"Lignes: {len(merged):,} | Colonnes: {len(merged.columns):,}")
        st.dataframe(merged.head(50), use_container_width=True)

    st.divider()

    # -------------------------
    # Chat state
    # -------------------------
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Salut 👋 Pose-moi une question sur les ventes, régions, catégories...\n\n"
                ),
            }
        ]

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # SQL (affiché seulement si toggle ON)
            if show_sql and msg.get("sql"):
                with st.expander("SQL généré", expanded=False):
                    st.code(msg["sql"], language="sql")

    user_q = st.chat_input(
        "Ex: Top 5 régions par ventes, prévision des ventes dans 30 jours pour la région Southeast..."
    )

    # -------------------------
    # Helpers
    # -------------------------
    def _is_forecast_question(q: str) -> bool:
        ql = q.lower()
        if any(k in ql for k in ["prévision", "prevision", "forecast", "prédire", "predire", "prediction"]):
            return True
        if re.search(r"\d+\s*(jour|jours|day|days)", ql) and any(k in ql for k in ["dans", "prochain", "next", "d'ici", "horizon"]):
            return True
        return False

    def _extract_horizon(q: str) -> int:
        m = re.search(r"(\d+)\s*(jour|jours|day|days)", q.lower())
        if not m:
            return int(default_horizon)
        n = int(m.group(1))
        return max(1, min(n, 365))

    def _auto_chart_sql(df: pd.DataFrame):
        if df is None or df.empty:
            return None, None

        cols = df.columns.tolist()
        date_col = None
        for c in cols:
            if "date" in c.lower() or "day" in c.lower():
                date_col = c
                break

        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cols if c not in num_cols]

        # date + numeric -> line
        if date_col and num_cols:
            y = num_cols[0]
            fig = px.line(df.sort_values(date_col), x=date_col, y=y)
            return fig, f"line: x={date_col}, y={y}"

        # category + numeric -> bar
        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            x = num_cols[0]
            y = cat_cols[0]
            fig = px.bar(df, x=x, y=y, orientation="h")
            return fig, f"bar: x={x}, y={y}"

        return None, "text_only"

    # -------------------------
    # Main
    # -------------------------
    if user_q:
        st.session_state["chat_messages"].append({"role": "user", "content": user_q})

        # ==========================================================
        # PATH 1 — Forecast ML
        # ==========================================================
        if _is_forecast_question(user_q):
            try:
                with st.spinner("🔮 Calcul de la prévision (ML)..."):
                    fr = forecast_sales(ROOT, user_q)  # horizon + region détectés dedans

                scope_label = "Global" if fr.scope == "global" else f"Région: {fr.region}"

                header = (
                    f"### Prévision ML — {scope_label}\n"
                    f"- Horizon: **{fr.horizon_days} jours**\n"
                    f"- Date cible: **{fr.forecast_date.date()}**\n"
                    f"- Ventes prévues à J+{fr.horizon_days}: **{fr.y_at_horizon:,.0f}**\n"
                    f"- Total prévu sur l’horizon: **{fr.future_sum:,.0f}**\n"
                )

                out = fr.df.copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=out["Date"], y=out["y_true"], mode="lines", name="Historique"))
                fig.add_trace(go.Scatter(x=out["Date"], y=out["y_pred"], mode="lines", name="Prévision"))
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)

                with st.spinner("📝 Explication (LLM)..."):
                    df_exp = pd.DataFrame([{
                        "scope": fr.scope,
                        "region": fr.region or "GLOBAL",
                        "horizon_days": fr.horizon_days,
                        "forecast_date": str(fr.forecast_date.date()),
                        "y_at_horizon": fr.y_at_horizon,
                        "future_sum": fr.future_sum,
                    }])

                    explanation = explain_result(
                        question=user_q,
                        sql="FORECAST_ML",
                        df=df_exp,
                        chart_hint="forecast_line",
                    )

                with st.chat_message("assistant"):
                    st.markdown(header)
                    st.markdown(explanation)
                    #st.subheader("Graphique")
                    #st.plotly_chart(fig, use_container_width=True)
                    #st.subheader("Détail prévision")
                    #st.dataframe(out.tail(fr.horizon_days + 30), use_container_width=True)

                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": header + "\n\n" + explanation}
                )
                st.rerun()

            except Exception as e:
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": f"❌ Erreur Forecast ML: {e}"}
                )
                st.rerun()

        # ==========================================================
        # PATH 2 — Analytics SQL
        # ==========================================================
        try:
            with st.spinner("🧠 Génération du SQL (Analytics)..."):
                sql = generate_sql(user_q, schema_hint=SCHEMA_HINT)

            if sql.strip().upper() == "UNSUPPORTED":
                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Je ne peux pas répondre à cette question. \n\n"
                            "Essaie une question sur: ventes, régions, catégories, dates."
                        ),
                    }
                )
                st.rerun()

            with st.spinner("🧮 Exécution SQL..."):
                qr = run_query(sql)
                df = qr.df

        except UnsafeSQLError as e:
            st.session_state["chat_messages"].append({"role": "assistant", "content": f"SQL rejeté (sécurité): {e}"})
            st.rerun()
        except Exception as e:
            st.session_state["chat_messages"].append({"role": "assistant", "content": f"Erreur d'exécution SQL: {e}"})
            st.rerun()

        fig, chart_hint = _auto_chart_sql(df)

        with st.spinner("📝 Explication (LLM)..."):
            explanation = explain_result(question=user_q, sql=sql, df=df, chart_hint=chart_hint)

        with st.chat_message("assistant"):
            if show_sql:
                with st.expander("SQL généré", expanded=False):
                    st.code(sql, language="sql")

            st.markdown(explanation)

            #st.subheader("Résultat")
            #st.dataframe(df.head(max_rows), use_container_width=True)

            #if fig is not None:
                #st.subheader("Graphique")
                #fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
                #st.plotly_chart(fig, use_container_width=True)

        st.session_state["chat_messages"].append({"role": "assistant", "content": explanation, "sql": sql})

        st.rerun()
