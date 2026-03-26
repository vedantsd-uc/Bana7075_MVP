
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------
# Project paths (robust & portable)
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "train.csv"
TEST_PATH = PROJECT_ROOT / "test.csv"


# ------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ------------------------------------------------------
# 2. Validate Schema & Quality
# ------------------------------------------------------
def validate_data(df: pd.DataFrame, is_train: bool = True) -> None:
    required_columns = {
        "datetime", "season", "holiday", "workingday",
        "weather", "temp", "atemp", "humidity", "windspeed"
    }

    if is_train:
        required_columns.add("count")

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if is_train and (df["count"] < 0).any():
        raise ValueError("Negative bike counts detected")

    if not df["datetime"].is_unique:
        raise ValueError("Duplicate datetime values detected")


# ------------------------------------------------------
# 3. Feature Engineering
# ------------------------------------------------------

# Machine‑learning models don’t understand timestamps directly.
# They do understand numeric patterns like “demand is higher on weekdays” or 
# “evenings are busier”.
# This function converts a timestamp into those signals.

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

# Bike demand at time t is often strongly related to demand at time t‑1.
# Lag features encode this “momentum” effect.

def add_lag_feature(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.sort_values("datetime")

    if is_train:
        df["lag_1_hour"] = df["count"].shift(1)
        df = df.dropna()
    else:
        df["lag_1_hour"] = 0  # placeholder for inference

    return df


# ------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------

# Numeric (e.g., temperature, hour)
# Categorical (e.g., season, weather)
# This function builds a single, unified preprocessing object that:
# scales numeric features,
# one‑hot encodes categorical features,
# applies the same logic to train and test data.

def build_preprocessor():
    categorical_features = [
        "season", "holiday", "workingday", "weather", "is_weekend"
    ]

    numeric_features = [
        "temp", "atemp", "humidity", "windspeed",
        "hour", "dayofweek", "month", "lag_1_hour"
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )


# ------------------------------------------------------
# 5. Full Pipeline
# ------------------------------------------------------

# Numerical Features temp, atemp (feels like), humidity, windspeed, hour, dayofweek, month, lag_1_hour (Bike count from previous hour)
# Categorical Features season (1,2,3,4 - 4 seasons), holiday (0,1 binary is it a holiday), 
#             workingday (0,1 is it a working day),  is_weekend (0,1 is it weekend) 
#             weather (1 - Clear, clear to partly cloudy, 2 - Mist, cloudy, 3 - Light rain or snow, 4 - Heavy rain or snow, severe conditions)                            
# Total Numerical Features - 8
# Total Categorical Features - 14 (sum up all the values, 4 seaons, 2 binary for holiday, 2 binary for working day, 2 binary for week end, 4 weather conditions)
# Total Features: 8 + 14 = 22
def run_training_pipeline():
    df = load_data(TRAIN_PATH)
    validate_data(df, is_train=True)

    df = add_time_features(df)
    df = add_lag_feature(df, is_train=True)

    y = df["count"].values
    X = df.drop(columns=["count", "datetime"])

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def run_inference_pipeline(preprocessor):
    df = load_data(TEST_PATH)
    validate_data(df, is_train=False)

    df = add_time_features(df)
    df = add_lag_feature(df, is_train=False)

    X = df.drop(columns=["datetime"])
    X_processed = preprocessor.transform(X)

    return X_processed

