"""NASA C-MAPSS turbofan Remaining Useful Life prediction pipeline."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
DROP_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}


def read_cmapss(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def add_rul(df, cap=125):
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycles - df["cycle"]).clip(upper=cap)
    return df


def engineer_features(df, window=10):
    df = df.copy().sort_values(["unit", "cycle"])
    sensor_cols = [c for c in df.columns if c.startswith("s") and c not in DROP_SENSORS]
    variable = [c for c in sensor_cols if df[c].std() >= 0.01]
    features = ["cycle", "os1", "os2", "os3"] + variable
    for col in variable:
        grouped = df.groupby("unit")[col]
        df[f"{col}_roll_mean"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{col}_roll_std"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
        features += [f"{col}_roll_mean", f"{col}_roll_std"]
    return df, features


def health_status(rul):
    if rul >= 90: return "Healthy"
    if rul >= 50: return "Monitor"
    if rul >= 20: return "Warning"
    return "Critical"


def train_and_evaluate(train_path, test_path=None, rul_path=None, cap=125):
    train_df = add_rul(read_cmapss(train_path), cap)
    train_df, features = engineer_features(train_df)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(train_df[features])
    y = train_df["RUL"]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    for name, model in models.items():
        model.fit(X, y)
        print(f"{name}: trained")
    if test_path and rul_path:
        test_df, _ = engineer_features(read_cmapss(test_path))
        last = test_df.groupby("unit", as_index=False).tail(1)
        X_test = scaler.transform(last[features])
        true_rul = pd.read_csv(rul_path, header=None).iloc[:, 0].to_numpy()
        for name, model in models.items():
            pred = model.predict(X_test)
            rmse = mean_squared_error(true_rul[:len(pred)], pred, squared=False)
            print(f"{name} RMSE: {rmse:.2f}")
        rf_pred = models["Random Forest"].predict(X_test)
        result = pd.DataFrame({"unit": last["unit"].values, "predicted_rul": rf_pred})
        result["health_status"] = result["predicted_rul"].map(health_status)
        return result
    return models


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test")
    p.add_argument("--rul")
    p.add_argument("--cap", type=int, default=125)
    args = p.parse_args()
    result = train_and_evaluate(args.train, args.test, args.rul, args.cap)
    if isinstance(result, pd.DataFrame):
        print(result.to_string(index=False))
