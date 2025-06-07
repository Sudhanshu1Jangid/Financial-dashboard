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

def plot_calendar_heatmap(df):
    """Plot daily spending heatmap for a given year/month."""
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    pivot = daily.pivot_table(index="day", columns="month", values="amount", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(pivot, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([pd.to_datetime(m, format="%m").strftime("%b") for m in range(1, 13)])
    ax.set_yticks(np.arange(1, 32))
    ax.set_yticklabels(range(1, 32))
    ax.set_xlabel("Month")
    ax.set_ylabel("Day")
    ax.set_title("Daily Spending Heatmap")
    fig.colorbar(cax, ax=ax, label="Spend (€)")
    st.pyplot(fig)

st.sidebar.header("Upload & Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload transactions file (CSV or Excel)", type=["csv", "xls", "xlsx"]
)

# Dynamic column mapping: show dropdowns after file is loaded
if uploaded_file:
    df_raw = load_file(uploaded_file)
    # Let user select which columns correspond
    st.sidebar.subheader("Map Columns")
    columns = df_raw.columns.tolist()
    date_col = st.sidebar.selectbox("Date Column", options=columns, index=columns.index("date") if "date" in columns else 0)
    desc_col = st.sidebar.selectbox("Description Column", options=columns, index=columns.index("description") if "description" in columns else 1)
    amt_col = st.sidebar.selectbox("Amount Column", options=columns, index=columns.index("amount") if "amount" in columns else 2)
else:
    date_col = desc_col = amt_col = None

# Category rules via JSON upload (optional)
st.sidebar.subheader("Category Rules (Optional)")
json_file = st.sidebar.file_uploader("Upload JSON for categories", type=["json"])
category_rules = load_category_rules(json_file) if uploaded_file else {}

if uploaded_file:
    df = parse_and_clean(df_raw, date_col, desc_col, amt_col)
    df = apply_categories(df, category_rules)
    monthly = compute_monthly(df)
    forecast = forecast_spending(monthly)
    st.sidebar.success("✅ Data processed successfully.")

if not uploaded_file:
    st.info("🔄 Please upload a transactions file to get started.")
else:
    tabs = st.tabs(["Overview", "Categories", "Transactions", "Forecast", "Heatmap", "Settings"])

    # 4.1 Overview Tab
    with tabs[0]:
        st.header("📑 Overview")
        col1, col2, col3 = st.columns(3)
        total_spent = df["amount"].sum()
        total_tx = len(df)
        date_min = df["date"].min().strftime("%Y-%m-%d")
        date_max = df["date"].max().strftime("%Y-%m-%d")
        col1.metric("Total Transactions", total_tx)
        col2.metric("Total Spent (€)", f"{total_spent:,.2f}")
        col3.metric("Date Range", f"{date_min} to {date_max}")

        st.subheader("Monthly Spending Trend")
        chart = (
            alt.Chart(monthly)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("amount:Q", title="Total Spent (€)"),
                tooltip=[alt.Tooltip("month:T", title="Month"), alt.Tooltip("amount:Q", title="Spent")],
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    # 4.2 Categories Tab
    with tabs[1]:
        st.header("🍽️ Spending by Category")
        cat_df = df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
        max_n = min(len(cat_df), 15)
        top_n = st.slider("Select Top N Categories", min_value=3, max_value=max_n, value=5)
        top_cats = cat_df.head(top_n)

        bar = (
            alt.Chart(top_cats)
            .mark_bar(color="#2ca02c")
            .encode(
                x=alt.X("amount:Q", title="Total Spent "),
                y=alt.Y("category:N", sort="-x", title="Category"),
                tooltip=[alt.Tooltip("category:N"), alt.Tooltip("amount:Q", title="€")],
            )
            .properties(width=700, height=350)
        )
        st.altair_chart(bar, use_container_width=True)

        st.subheader("Category Breakdown by Month")
        selected_cat = st.multiselect("Choose categories to compare", options=cat_df["category"].tolist(), default=cat_df["category"].tolist()[:3])
        if selected_cat:
            cat_month = (
                df[df["category"].isin(selected_cat)]
                .groupby(["month", "category"])["amount"]
                .sum()
                .reset_index()
            )
            line = (
                alt.Chart(cat_month)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("amount:Q", title="Spending (€)"),
                    color=alt.Color("category:N"),
                    tooltip=["month:T", "category:N", "amount:Q"],
                )
                .properties(width=800, height=400)
            )
            st.altair_chart(line, use_container_width=True)

    # 4.3 Transactions Tab
    with tabs[2]:
        st.header("🔍 Inspect Transactions")
        with st.expander("Filter Options"):
            cats = st.multiselect("Filter by Category", options=sorted(df["category"].unique()), default=[])
            drange = st.date_input("Filter by Date Range", value=(df["date"].min(), df["date"].max()))
        mask = pd.Series(True, index=df.index)
        if cats:
            mask &= df["category"].isin(cats)
        start_date, end_date = drange
        mask &= (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
        filtered = df[mask].sort_values("date", ascending=False)
        st.dataframe(filtered.reset_index(drop=True))

    # 4.4 Forecast Tab
    with tabs[3]:
        st.header("🔮 Forecast Next Month Spending")
        if not forecast:
            st.warning("At least 6 months of data required for forecast.")
        else:
            st.metric("Mean Absolute Error (Test)", f"€ {forecast['mae']:,.2f}")
            nm = forecast["next_month"].strftime("%b %Y")
            st.metric(f"Predicted Spending {nm}", f"€ {forecast['next_pred']:,.2f}")

            df_pred = forecast["df_pred"]
            source = pd.melt(df_pred, id_vars=["month"], value_vars=["actual", "predicted"],
                              var_name="Type", value_name="€ Spent")
            fc_chart = (
                alt.Chart(source)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y(" Spent:Q"),
                    color="Type:N",
                    tooltip=["month:T", "Type:N", "€ Spent:Q"],
                )
                .properties(width=800, height=400)
            )
            st.altair_chart(fc_chart, use_container_width=True)

    # 4.5 Heatmap Tab
    with tabs[4]:
        st.header("📅 Daily Spending Heatmap")
        plot_calendar_heatmap(df)

    # 4.6 Settings Tab
    with tabs[5]:
        st.header("⚙️ Settings & Info")
        st.markdown(
            """
- **Category Rules JSON**: Upload a JSON file with structure:
```json
{
  "CategoryName1": ["keyword1", "keyword2"],
  "CategoryName2": ["keywordA", "keywordB"]
}""")