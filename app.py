import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LIFTS = {"Squat": "Best3SquatKg", "Bench": "Best3BenchKg", "Deadlift": "Best3DeadliftKg"}
FEATURES = ["Age", "BodyweightKg", "log_BodyweightKg", "Age_sq"]


@st.cache_data
def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("brianmcabee/us-powerlifting-competition-data-2015now")
    df = pd.read_csv(f"{path}/usa_sbd_data_2020-10-16.csv")
    df = df[df["Federation"] == "USPA"]
    df = df.dropna(subset=["Age", "BodyweightKg", "Sex", "Best3BenchKg", "Best3SquatKg", "Best3DeadliftKg"])
    df = df[(df["Best3SquatKg"] > 0) & (df["Best3BenchKg"] > 0) & (df["Best3DeadliftKg"] > 0)]
    df["log_BodyweightKg"] = np.log(df["BodyweightKg"])
    df["Age_sq"] = df["Age"] ** 2
    df["bench_ratio"] = df["Best3BenchKg"] / df["BodyweightKg"]
    df["squat_ratio"] = df["Best3SquatKg"] / df["BodyweightKg"]
    df["deadlift_ratio"] = df["Best3DeadliftKg"] / df["BodyweightKg"]
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), FEATURES), ("cat", OneHotEncoder(drop="first"), ["Sex"])]
    )
    models, metrics = {}, {}
    X = df[FEATURES + ["Sex"]]
    for lift_name, col in LIFTS.items():
        y = np.log(df[col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Pipeline([("pre", preprocessor), ("lr", LinearRegression())])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[lift_name] = model
        metrics[lift_name] = {
            "r2": r2_score(y_test, pred),
            "mae_kg": mean_absolute_error(np.exp(y_test), np.exp(pred)),
            "rmse_kg": root_mean_squared_error(np.exp(y_test), np.exp(pred)),
        }
    return models, metrics


def predict(models, age: float, bodyweight: float, sex: str) -> dict:
    row = pd.DataFrame(
        [{"Age": age, "BodyweightKg": bodyweight, "log_BodyweightKg": np.log(bodyweight), "Age_sq": age**2, "Sex": sex}]
    )
    return {lift: float(np.exp(model.predict(row)[0])) for lift, model in models.items()}


st.set_page_config(page_title="SBD Powerlifting Predictor", page_icon="🏋️", layout="wide")
st.title("🏋️ SBD Powerlifting Total Predictor")
st.caption("Trained on USPA competition results (2015–present) via Kaggle dataset `us-powerlifting-competition-data-2015now`.")

df = load_data()
models, metrics = train_models(df)

tab_predict, tab_explore, tab_model = st.tabs(["Predict", "Explore the Data", "Model Performance"])

with tab_predict:
    col1, col2 = st.columns([1, 2])
    with col1:
        age = st.slider("Age", 13, 80, 25)
        bodyweight = st.slider("Bodyweight (kg)", 40.0, 180.0, 80.0, step=0.5)
        sex = st.radio("Sex", ["M", "F"], horizontal=True)
        run = st.button("Predict my lifts", type="primary")

    with col2:
        if run:
            preds = predict(models, age, bodyweight, sex)
            total = sum(preds.values())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Squat", f"{preds['Squat']:.1f} kg")
            c2.metric("Bench", f"{preds['Bench']:.1f} kg")
            c3.metric("Deadlift", f"{preds['Deadlift']:.1f} kg")
            c4.metric("Total", f"{total:.1f} kg")
        else:
            st.info("Set your stats and click **Predict my lifts**.")

with tab_explore:
    df_m = df[(df["Sex"] == "M") & (df["bench_ratio"] <= 2) & (df["squat_ratio"] <= 3) & (df["deadlift_ratio"] <= 3.5)]
    df_f = df[(df["Sex"] == "F") & (df["bench_ratio"] <= 1.5) & (df["squat_ratio"] <= 2.5) & (df["deadlift_ratio"] <= 3)]

    st.subheader("Lift vs. Bodyweight")
    lift_cols = st.columns(3)
    for col, (lift_name, lift_col) in zip(lift_cols, LIFTS.items()):
        with col:
            fig, ax = plt.subplots()
            ax.scatter(df_m[lift_col], df_m["BodyweightKg"], alpha=0.2, color="cyan", label="M")
            ax.scatter(df_f[lift_col], df_f["BodyweightKg"], alpha=0.2, color="pink", label="F")
            ax.set_xlabel(lift_col)
            ax.set_ylabel("BodyweightKg")
            ax.legend()
            st.pyplot(fig)

    st.subheader("Lift vs. Age")
    age_cols = st.columns(3)
    for col, (lift_name, lift_col) in zip(age_cols, LIFTS.items()):
        with col:
            fig, ax = plt.subplots()
            ax.scatter(df_m["Age"], df_m[lift_col], alpha=0.2, color="cyan", label="M")
            ax.scatter(df_f["Age"], df_f[lift_col], alpha=0.2, color="pink", label="F")
            ax.set_xlabel("Age")
            ax.set_ylabel(lift_col)
            ax.legend()
            st.pyplot(fig)

with tab_model:
    st.subheader("Final model: Linear Regression on log(lift)")
    st.write("Features: `Age`, `BodyweightKg`, `log(BodyweightKg)`, `Age²`, `Sex` (one-hot).")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format({"r2": "{:.3f}", "mae_kg": "{:.1f}", "rmse_kg": "{:.1f}"}))
