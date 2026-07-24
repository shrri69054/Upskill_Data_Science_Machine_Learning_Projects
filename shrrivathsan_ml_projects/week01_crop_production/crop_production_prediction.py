"""Agriculture crop production prediction."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLUMNS = ["Crop", "Variety", "State", "Season", "Recommended Zone"]
TARGET = "Production"


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path).drop_duplicates().dropna().copy()
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            values = {v: i for i, v in enumerate(sorted(df[col].astype(str).unique()))}
            encoders[col] = values
            df[col] = df[col].astype(str).map(values).astype(int)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y, encoders


def train(csv_path: str, model_path: str = "models/crop_production_rf.joblib"):
    X, y, encoders = load_and_prepare(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, pred):.4f}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders, "columns": list(X.columns)}, model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datafile.csv")
    parser.add_argument("--model", default="models/crop_production_rf.joblib")
    args = parser.parse_args()
    train(args.data, args.model)
