"""
Intelligent Sales Forecasting Dashboard
----------------------------------------
Streamlit app for the Superstore Sales Forecasting project.
Only requires train.csv to run (fully self-contained - safe to deploy as-is).

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# =====================================================================
# DATA LOADING (cached so it only runs once)
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Season"] = df["Order Month"].apply(
        lambda m: "Winter" if m in [12, 1, 2] else "Spring" if m in [3, 4, 5]
        else "Summer" if m in [6, 7, 8] else "Autumn"
    )
    return df


def season_num(m):
    return 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3


FEATURES = ["lag1", "lag2", "lag3", "rolling_mean_3", "month", "quarter", "season"]


def make_monthly_series(sub_df):
    s = sub_df.set_index("Order Date").resample("MS")["Sales"].sum()
    return s.asfreq("MS").fillna(0)


def build_lag_features(monthly_series):
    ml = monthly_series.reset_index()
    ml.columns = ["ds", "y"]
    ml["lag1"] = ml["y"].shift(1)
    ml["lag2"] = ml["y"].shift(2)
    ml["lag3"] = ml["y"].shift(3)
    ml["rolling_mean_3"] = ml["y"].shift(1).rolling(3).mean()
    ml["month"] = ml["ds"].dt.month
    ml["quarter"] = ml["ds"].dt.quarter
    ml["season"] = ml["month"].apply(season_num)
    return ml.dropna().reset_index(drop=True)


@st.cache_data
def evaluate_xgboost(monthly_series):
    """Train/test split evaluation (last 3 months held out) -> MAE, RMSE."""
    ml = build_lag_features(monthly_series)
    train, test = ml.iloc[:-3], ml.iloc[-3:]
    if len(train) < 5:
        return None, None
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[FEATURES], train["y"])
    pred = model.predict(test[FEATURES])
    mae = float(np.mean(np.abs(test["y"].values - pred)))
    rmse = float(np.sqrt(np.mean((test["y"].values - pred) ** 2)))
    return mae, rmse


@st.cache_data
def forecast_xgboost(monthly_series, periods):
    """Recursive future forecast using the full history."""
    ml = build_lag_features(monthly_series)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(ml[FEATURES], ml["y"])

    history = list(monthly_series.values)
    future_dates = pd.date_range(monthly_series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    preds = []
    for d in future_dates:
        lag1, lag2, lag3 = history[-1], history[-2], history[-3]
        roll3 = np.mean(history[-3:])
        row = pd.DataFrame([[lag1, lag2, lag3, roll3, d.month, d.quarter, season_num(d.month)]], columns=FEATURES)
        p = model.predict(row)[0]
        preds.append(p)
        history.append(p)
    return future_dates, preds


@st.cache_data
def run_anomaly_detection(df):
    weekly = df.set_index("Order Date").resample("W")["Sales"].sum().asfreq("W").fillna(0).to_frame("Weekly Sales")

    iso = IsolationForest(contamination=0.06, random_state=42)
    weekly["iso_flag"] = iso.fit_predict(weekly[["Weekly Sales"]]) == -1

    window = 6
    roll_mean = weekly["Weekly Sales"].rolling(window, min_periods=1).mean()
    roll_std = weekly["Weekly Sales"].rolling(window, min_periods=1).std().fillna(0)
    weekly["z_score"] = (weekly["Weekly Sales"] - roll_mean) / roll_std.replace(0, np.nan)
    weekly["zscore_flag"] = weekly["z_score"].abs() > 2
    return weekly


@st.cache_data
def run_clustering(df):
    rows = []
    for sub_cat, g in df.groupby("Sub-Category"):
        total_sales = g["Sales"].sum()
        avg_order_value = g.groupby("Order ID")["Sales"].sum().mean()
        monthly_g = g.set_index("Order Date").resample("MS")["Sales"].sum().reindex(
            pd.date_range(df["Order Date"].min(), df["Order Date"].max(), freq="MS"), fill_value=0)
        volatility = monthly_g.std()
        yearly = g.groupby("Order Year")["Sales"].sum()
        growth_rate = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
                        if len(yearly) >= 2 and yearly.iloc[0] > 0 else 0)
        rows.append({"Sub-Category": sub_cat, "Total Sales": total_sales, "Growth Rate (%)": growth_rate,
                     "Volatility": volatility, "Avg Order Value": avg_order_value})
    feat_df = pd.DataFrame(rows)
    cols = ["Total Sales", "Growth Rate (%)", "Volatility", "Avg Order Value"]
    X_scaled = StandardScaler().fit_transform(feat_df[cols].values)

    k = 4
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat_df["Cluster"] = km.fit_predict(X_scaled)
    profile = feat_df.groupby("Cluster")[cols].mean()

    def label_cluster(row):
        if row["Total Sales"] >= profile["Total Sales"].median() and row["Volatility"] <= profile["Volatility"].median():
            return "High Volume, Stable Demand"
        elif row["Growth Rate (%)"] > 20:
            return "Growing Demand"
        elif row["Growth Rate (%)"] < -10:
            return "Declining Demand"
        else:
            return "Low Volume, High Volatility"

    labels = {c: label_cluster(profile.loc[c]) for c in profile.index}
    feat_df["Segment Label"] = feat_df["Cluster"].map(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    feat_df["PCA1"], feat_df["PCA2"] = coords[:, 0], coords[:, 1]
    return feat_df


# =====================================================================
# LOAD DATA
# =====================================================================
df = load_data()

st.sidebar.title("📦 Sales Forecasting")
page = st.sidebar.radio("Go to", ["1. Sales Overview", "2. Forecast Explorer", "3. Anomaly Report", "4. Product Segments"])

# =====================================================================
# PAGE 1 - SALES OVERVIEW
# =====================================================================
if page == "1. Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly = df.groupby("Order Year")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(yearly.index.astype(str), yearly.values, color="#2563eb")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = df.set_index("Order Date").resample("MS")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(monthly.index, monthly.values, color="#16a34a", marker="o", markersize=3)
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    st.subheader("Sales by Region & Category (interactive filters)")
    c1, c2 = st.columns(2)
    with c1:
        region_filter = st.multiselect("Filter by Region", options=sorted(df["Region"].unique()),
                                        default=sorted(df["Region"].unique()))
    with c2:
        category_filter = st.multiselect("Filter by Category", options=sorted(df["Category"].unique()),
                                          default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    pivot = filtered.groupby(["Region", "Category"])["Sales"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylabel("Sales ($)")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.dataframe(pivot.style.format("{:,.0f}"))

# =====================================================================
# PAGE 2 - FORECAST EXPLORER
# =====================================================================
elif page == "2. Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption("Forecasts are generated with the XGBoost model, the best-performing model from the notebook's Task 3 comparison.")

    dim_type = st.selectbox("Select dimension type", ["Category", "Region"])
    if dim_type == "Category":
        options = sorted(df["Category"].unique())
    else:
        options = sorted(df["Region"].unique())
    dim_value = st.selectbox(f"Select {dim_type}", options)

    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    sub_df = df[df[dim_type] == dim_value]
    monthly_series = make_monthly_series(sub_df)

    mae, rmse = evaluate_xgboost(monthly_series)
    future_dates, preds = forecast_xgboost(monthly_series, horizon)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_series.index, monthly_series.values, label="Actual", color="black")
    ax.plot(future_dates, preds, label="Forecast", color="#dc2626", marker="o", linestyle="--")
    ax.set_title(f"{horizon}-Month Forecast — {dim_value} ({dim_type})")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecasted values")
    forecast_table = pd.DataFrame({"Month": [d.strftime("%B %Y") for d in future_dates],
                                    "Forecasted Sales ($)": [round(p, 2) for p in preds]})
    st.table(forecast_table)

    st.subheader("Model accuracy (evaluated on last 3 known months)")
    m1, m2 = st.columns(2)
    if mae is not None:
        m1.metric("MAE", f"${mae:,.2f}")
        m2.metric("RMSE", f"${rmse:,.2f}")
    else:
        st.info("Not enough history in this segment to compute a held-out accuracy score.")

# =====================================================================
# PAGE 3 - ANOMALY REPORT
# =====================================================================
elif page == "3. Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.caption("Weekly sales anomalies detected using Isolation Forest and a rolling Z-score (>2 std from a 6-week rolling mean).")

    weekly = run_anomaly_detection(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly.index, weekly["Weekly Sales"], color="#334155", linewidth=1, label="Weekly Sales")
    ax.scatter(weekly[weekly["iso_flag"]].index, weekly[weekly["iso_flag"]]["Weekly Sales"],
               color="#dc2626", s=60, label="Isolation Forest Anomaly", zorder=5)
    ax.scatter(weekly[weekly["zscore_flag"]].index, weekly[weekly["zscore_flag"]]["Weekly Sales"],
               color="#f59e0b", s=130, marker="x", label="Z-score Anomaly", zorder=4)
    ax.set_ylabel("Sales ($)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    anomaly_table = weekly[weekly["iso_flag"] | weekly["zscore_flag"]][
        ["Weekly Sales", "iso_flag", "zscore_flag", "z_score"]
    ].reset_index()
    anomaly_table.columns = ["Week", "Sales ($)", "Flagged by Isolation Forest", "Flagged by Z-score", "Z-score"]
    st.dataframe(anomaly_table.style.format({"Sales ($)": "{:,.2f}", "Z-score": "{:,.2f}"}))

# =====================================================================
# PAGE 4 - PRODUCT DEMAND SEGMENTS
# =====================================================================
elif page == "4. Product Segments":
    st.title("🧩 Product Demand Segments")
    st.caption("Sub-categories clustered by total sales, growth rate, volatility, and average order value (K-Means, k=4).")

    feat_df = run_clustering(df)

    palette = {"High Volume, Stable Demand": "#2563eb", "Growing Demand": "#16a34a",
               "Declining Demand": "#dc2626", "Low Volume, High Volatility": "#f59e0b"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, group in feat_df.groupby("Segment Label"):
        ax.scatter(group["PCA1"], group["PCA2"], label=label, s=100,
                   color=palette.get(label, "#7c3aed"), edgecolor="black")
        for _, r in group.iterrows():
            ax.annotate(r["Sub-Category"], (r["PCA1"], r["PCA2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-categories by demand segment")
    display_df = feat_df[["Sub-Category", "Segment Label", "Total Sales", "Growth Rate (%)", "Volatility", "Avg Order Value"]]
    st.dataframe(display_df.style.format({
        "Total Sales": "{:,.0f}", "Growth Rate (%)": "{:,.1f}",
        "Volatility": "{:,.0f}", "Avg Order Value": "{:,.2f}"
    }))

    st.subheader("Recommended stocking strategy")
    st.markdown("""
- **High Volume, Stable Demand**: Keep continuous safety stock, automate reordering.
- **Growing Demand**: Gradually increase order quantities; monitor monthly trend before over-committing capital.
- **Declining Demand**: Reduce standing inventory, move to just-in-time ordering.
- **Low Volume, High Volatility**: Order in small batches on demand rather than holding stock.
""")
