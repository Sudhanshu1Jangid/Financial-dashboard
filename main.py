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

