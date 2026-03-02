from __future__ import annotations
import pandas as pd
import numpy as np

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Macro category 
    if "product_category_name" in out.columns:
        TECH_ELECTRONICS = {
            "computers_accessories","electronics","audio","consoles_games",
            "tables_printing_image","computers"
        }
        HOME_FURNITURE = {
            "garden_tools","bed_bath_table","furniture_decor","office_furniture",
            "housewares","home_appliances","home_appliances_2",
            "kitchen_dining_laundry_garden_furniture","furniture_living_room",
            "furniture_bedroom","home_confort","home_comfort_2"
        }
        FASHION_BEAUTY = {
            "health_beauty","watches_gifts","perfumery","fashion_bags_accessories",
            "fashion_shoes","fashion_sport","fashion_childrens_clothes",
            "fashio_female_clothing","fashion_male_clothing","fashion_underwear_beach"
        }
        LEISURE_ENT = {
            "toys","cool_stuff","sports_leisure","books_technical",
            "books_general_interest","books_imported","musical_instruments",
            "dvds_blu_ray","cine_photo","music"
        }
        EVERYDAY_MISC = {
            "stationery","market_place","small_appliances","party_supplies",
            "arts_and_craftmanship","pet_shop","baby"
        }

        def map_macro(x: str) -> str:
            if x in TECH_ELECTRONICS: return "Tech & Electronics"
            if x in HOME_FURNITURE: return "Home & Furniture"
            if x in FASHION_BEAUTY: return "Fashion & Beauty"
            if x in LEISURE_ENT: return "Leisure & Entertainment"
            if x in EVERYDAY_MISC: return "Everyday & Misc"
            return "Others"

        out["Macro_Category"] = out["product_category_name"].astype(str).apply(map_macro)

    # Region mapping 
    if "customer_state" in out.columns:
        state_to_region = {
            "SP": "Southeast","RJ": "Southeast","MG": "Southeast","ES": "Southeast",
            "RS": "South","PR": "South","SC": "South",
            "BA": "Northeast","PE": "Northeast","CE": "Northeast","PB": "Northeast",
            "AL": "Northeast","RN": "Northeast","SE": "Northeast","MA": "Northeast","PI": "Northeast",
            "GO": "Central-West","DF": "Central-West","MT": "Central-West","MS": "Central-West",
            "PA": "North","AM": "North","RO": "North","TO": "North",
            "AP": "North","AC": "North","RR": "North",
        }
        out["Region"] = out["customer_state"].map(state_to_region).fillna("Unknown")

    return out

def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["Year"] = out[date_col].dt.year
    out["Month"] = out[date_col].dt.month
    out["Quarter"] = out[date_col].dt.quarter
    out["Day"] = out[date_col].dt.day
    out["DayOfWeek"] = out[date_col].dt.dayofweek
    out["DayOfYear"] = out[date_col].dt.dayofyear
    out["WeekOfYear"] = out[date_col].dt.isocalendar().week.astype(int)
    out["IsWeekend"] = out["DayOfWeek"].isin([5, 6]).astype(int)
    return out

def add_lags_and_rollings(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    lags=(1, 7, 14, 30),
    windows=(7, 14, 30),
) -> pd.DataFrame:
    out = df.sort_values([group_col, date_col]).copy()

    for lag in lags:
        out[f"sales_lag_{lag}"] = out.groupby(group_col)[target_col].shift(lag)

    for w in windows:
        g = out.groupby(group_col)[target_col]
        out[f"rolling_mean_{w}"] = g.shift(1).rolling(w).mean()
        out[f"rolling_std_{w}"] = g.shift(1).rolling(w).std()

    return out