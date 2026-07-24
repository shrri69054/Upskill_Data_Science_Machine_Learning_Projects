"""Smart City traffic forecasting with Gradient Boosting."""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score


def add_features(df):
    df = df.copy()
    date_col = "DateTime" if "DateTime" in df.columns else "Datetime"
    df[date_col] = pd.to_datetime(df[date_col])
    dt = df[date_col]
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak_hour"] = dt.dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 22)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def train(train_csv, target="Vehicles", junction_col="Junction"):
    df = add_features(pd.read_csv(train_csv))
    features = [c for c in df.columns if c not in {target, "DateTime", "Datetime", "ID"}]
    X = pd.get_dummies(df[features], columns=[junction_col], drop_first=False)
    y = df[target]
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"CV MAE: {scores.mean():.2f} ± {scores.std():.2f}")
    model.fit(X, y)
    return model, X.columns


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default="Vehicles")
    args = p.parse_args()
    train(args.data, args.target)
