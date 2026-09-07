 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
 
st.set_page_config(page_title="Rainfall → Runoff Predictions", layout="wide")
 
 
# ---------------------------------------------------------------------------
# Cached loaders — run once, reused across interactions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load("runoff_xgboost.pkl")
    lstm_model = load_model("runoff_lstm.keras")
    scaler_X = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return xgb_model, lstm_model, scaler_X, scaler_y
 
 
@st.cache_data
def load_and_engineer_features(seq_length: int):
    df = pd.read_csv("all_stations.csv")
 
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, format="mixed", errors="coerce")
        df = df.sort_values("fecha")
 
    df = df.dropna().reset_index(drop=True)
 
    location_cols = [c for c in df.columns if c not in ["fecha", "runoff", "total_rainfall"]]
    df["total_rainfall"] = df[location_cols].sum(axis=1)
 
    target_station = "Barcelona_fabra" if "Barcelona_fabra" in df.columns else location_cols[0]
    df["rainfall_lag1"] = df[target_station].shift(1)
    df["rainfall_lag2"] = df[target_station].shift(2)
    df["rainfall_lag3"] = df[target_station].shift(3)
 
    if "runoff" not in df.columns:
        df["runoff"] = df[target_station] * 0.4 + df["total_rainfall"] * 0.1
 
    df["runoff_lag1"] = df["runoff"].shift(1)
    df = df.dropna().reset_index(drop=True)
 
    feature_cols = ["total_rainfall", "rainfall_lag1", "rainfall_lag2", "rainfall_lag3", "runoff_lag1"]
    return df, feature_cols
 
 
def make_sequences(X_arr, y_arr, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X_arr) - seq_len):
        X_seq.append(X_arr[i:i + seq_len])
        y_seq.append(y_arr[i + seq_len])
    return np.array(X_seq), np.array(y_seq)
 
 
# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
st.title("🌧️ Rainfall → Runoff: Actual vs Predicted")
 
try:
    xgb_model, lstm_model, scaler_X, scaler_y = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Make sure runoff_xgboost.pkl, "
        "runoff_lstm.keras, scaler_x.pkl and scaler_y.pkl are in the same "
        "folder as app.py (run the training script first if you haven't)."
    )
    st.stop()
 
SEQ_LENGTH = 7  # must match the value used during training
df, feature_cols = load_and_engineer_features(SEQ_LENGTH)
 
# Scale full feature/target arrays using the SAME scalers fit during training
X_scaled = scaler_X.transform(df[feature_cols])
y_scaled = scaler_y.transform(df[["runoff"]])
 
X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LENGTH)
dates_seq = df["fecha"].iloc[SEQ_LENGTH:].reset_index(drop=True) if "fecha" in df.columns else None
 
# Same chronological 80/20 split as training
split = int(len(X_seq) * 0.8)
X_test_seq, y_test_seq = X_seq[split:], y_seq[split:]
X_test_last = X_test_seq[:, -1, :]  # last timestep, for the tree model
dates_test = dates_seq.iloc[split:].reset_index(drop=True) if dates_seq is not None else None
 
 
# ---------------------------------------------------------------------------
# Exploratory Data Analysis: rainfall by location, total rainfall, heatmap
# ---------------------------------------------------------------------------
station_cols = [
    c
    for c in df.columns
    if c not in ["fecha", "runoff", "total_rainfall", "rainfall_lag1", "rainfall_lag2", "rainfall_lag3", "runoff_lag1"]
]
 
with st.expander("Exploratory Data Analysis", expanded=False):
    st.markdown("**Rainfall over time by location**")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for col in station_cols:
        ax1.plot(df["fecha"], df[col], label=col)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Rainfall")
    ax1.set_title("Rainfall Over Time by Location")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    fig1.tight_layout()
    st.pyplot(fig1)
 
    st.markdown("**Total rainfall over time**")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df["fecha"], df["total_rainfall"], color="steelblue")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Rainfall")
    ax2.set_title("Total Rainfall Over Time")
    fig2.tight_layout()
    st.pyplot(fig2)
 
    st.markdown("**Runoff over time**")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(df["fecha"], df["runoff"], color="darkred")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Runoff")
    ax3.set_title("Runoff Over Time")
    fig3.tight_layout()
    st.pyplot(fig3)
 
    st.markdown("**Correlation heatmap**")
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation Between Features")
    fig4.tight_layout()
    st.pyplot(fig4)
 
 
# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Options")
model_choice = st.sidebar.radio("Model", ["XGBoost", "LSTM", "Both"], index=2)
n_points = st.sidebar.slider(
    "Points to show (most recent)", min_value=20, max_value=len(y_test_seq), value=min(200, len(y_test_seq))
)
 
 
# ---------------------------------------------------------------------------
# Predictions (inverse-transformed back to real runoff units)
# ---------------------------------------------------------------------------
y_test_real = scaler_y.inverse_transform(y_test_seq).flatten()
 
xgb_pred_real = scaler_y.inverse_transform(
    xgb_model.predict(X_test_last).reshape(-1, 1)
).flatten()
 
lstm_pred_real = scaler_y.inverse_transform(
    lstm_model.predict(X_test_seq, verbose=0).reshape(-1, 1)
).flatten()
 
# Slice to the most recent n_points selected in the sidebar
y_plot = y_test_real[-n_points:]
xgb_plot = xgb_pred_real[-n_points:]
lstm_plot = lstm_pred_real[-n_points:]
dates_plot = dates_test.iloc[-n_points:] if dates_test is not None else pd.RangeIndex(n_points)
 
 
# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_row(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }
 
 
rows = []
if model_choice in ("XGBoost", "Both"):
    rows.append(metrics_row("XGBoost", y_test_real, xgb_pred_real))
if model_choice in ("LSTM", "Both"):
    rows.append(metrics_row("LSTM", y_test_real, lstm_pred_real))
 
st.subheader("Test-set metrics")
st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
 
 
# ---------------------------------------------------------------------------
# Chart: Actual vs Predicted
# ---------------------------------------------------------------------------
st.subheader("Actual vs Predicted Runoff")
 
chart_df = pd.DataFrame({"Actual": y_plot}, index=dates_plot)
if model_choice in ("XGBoost", "Both"):
    chart_df["XGBoost"] = xgb_plot
if model_choice in ("LSTM", "Both"):
    chart_df["LSTM"] = lstm_plot
 
st.line_chart(chart_df, use_container_width=True)
 
 
# ---------------------------------------------------------------------------
# Raw rainfall data (optional context)
# ---------------------------------------------------------------------------
with st.expander("Show raw rainfall/runoff data"):
    st.dataframe(df.tail(200), use_container_width=True)
 