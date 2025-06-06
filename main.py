import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import json

st.set_page_config(
    page_title="Enhanced Finance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    """Load CSV or Excel into DataFrame."""
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        st.error("Failed to read file. Please upload a valid CSV or Excel file.")
        st.stop()

@st.cache_data(show_spinner=False)
def parse_and_clean(df, date_col, desc_col, amt_col):
    """Keep only needed columns, parse dates and amounts."""
    required = {date_col, desc_col, amt_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    df = df[[date_col, desc_col, amt_col]].copy()
    df.rename(columns={date_col: "date", desc_col: "description", amt_col: "amount"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    # Convert all expenses to positive values
    df["amount"] = df["amount"].abs()
    # Assign month period for grouping
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_data(show_spinner=False)
def load_category_rules(json_file):
    """Load category rules from uploaded JSON or use defaults."""
    if json_file is None:
        # Default keyword mapping
        # Rest you can add by yourself...
        return {
            "Groceries": ["supermarket", "grocery", "market", "aldi", "lidl", "tesco"],
            "Restaurants": ["cafe", "restaurant", "bar", "starbucks", "mcdonald", "domino"],
            "Rent": ["rent", "landlord", "leasing"],
            "Utilities": ["electricity", "water", "sewer", "gas bill", "internet", "telia", "vodafone"],
            "Transportation": ["uber", "lyft", "bus", "train", "taxi", "fuel", "gas station"],
            "Entertainment": ["netflix", "spotify", "cinema", "theatre", "concert"],
            "Healthcare": ["pharmacy", "hospital", "clinic", "doctor"],
            "Salary": ["salary", "payroll"],
        }
    try:
        rules = json.load(json_file)
        return rules
    except Exception:
        st.error("Invalid JSON for categories. Please upload a valid JSON file with format {\"Category\": [\"keyword1\", ...], ...}.")
        st.stop()

@st.cache_data(show_spinner=False)
def apply_categories(df, rules):
    """Categorize descriptions based on keyword rules."""
    def map_cat(desc):
        d = desc.lower()
        for cat, keywords in rules.items():
            for kw in keywords:
                if kw.lower() in d:
                    return cat
        return "Other"
    df["category"] = df["description"].apply(map_cat)
    return df

@st.cache_data(show_spinner=False)
def compute_monthly(df):
    """Aggregate total spending per month."""
    monthly = df.groupby("month")["amount"].sum().reset_index().sort_values("month")
    return monthly

@st.cache_data(show_spinner=False)
def forecast_spending(monthly):
    """Linear regression forecast for next month."""
    X = monthly["month"].map(lambda d: d.toordinal()).values.reshape(-1, 1)
    y = monthly["amount"].values
    if len(X) < 6:
        return None
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    last_month = monthly["month"].max()
    next_month_dt = (last_month + pd.DateOffset(months=1))
    next_month_num = next_month_dt.toordinal()
    next_pred = model.predict(np.array([[next_month_num]]))[0]
    df_pred = pd.DataFrame({
        "month": monthly["month"].iloc[split:],
        "actual": y_test,
        "predicted": y_pred_test
    })
    return {
        "mae": mae,
        "next_month": next_month_dt,
        "next_pred": next_pred,
        "df_pred": df_pred
    }