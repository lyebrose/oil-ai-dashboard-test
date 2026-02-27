# model.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.inspection import permutation_importance


FEATURE_COLS = [
    # Technicals
    "ret_1d", "ret_5d", "ret_20d",
    "vol_5d", "vol_20d", "vol_60d",
    "price_over_ma20", "price_over_ma50",
    "drawdown_20d",
    # GDELT
    "energy_news_count", "energy_news_tone", "news_spike_ratio",
]


@dataclass
class BacktestResult:
    preds: pd.Series
    truth: pd.Series
    mae: float
    rmse: float
    dir_acc: float
    importance: pd.DataFrame


def walk_forward_backtest(df: pd.DataFrame, start_year=2015) -> BacktestResult:
    """
    Walk-forward by year:
      Train up to year-1, test on that year.
    """
    df = df.copy()
    df["year"] = df.index.year

    preds_all = []
    truth_all = []

    # Keep last trained model for “importance”
    last_model = None
    last_X_test = None
    last_y_test = None

    years = sorted(y for y in df["year"].unique() if y >= start_year)
    for y in years:
        train = df[df["year"] < y]
        test = df[df["year"] == y]
        if len(train) < 300 or len(test) < 50:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train["y_week_return"]
        X_test = test[FEATURE_COLS]
        y_test = test["y_week_return"]

        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            random_state=42,
        )
        model.fit(X_train, y_train)

        pred = pd.Series(model.predict(X_test), index=X_test.index)
        preds_all.append(pred)
        truth_all.append(y_test)

        last_model = model
        last_X_test = X_test
        last_y_test = y_test

    preds = pd.concat(preds_all).sort_index()
    truth = pd.concat(truth_all).sort_index()

    mae = float(mean_absolute_error(truth, preds))
    rmse = float(np.sqrt(mean_squared_error(truth, preds)))

    # Direction accuracy on sign
    dir_acc = float(accuracy_score((truth > 0).astype(int), (preds > 0).astype(int)))

    # Permutation importance (on last test fold)
    importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": 0.0})
    if last_model is not None and last_X_test is not None and len(last_X_test) > 200:
        r = permutation_importance(
            last_model, last_X_test, last_y_test, n_repeats=10, random_state=42
        )
        importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": r.importances_mean})
        importance = importance.sort_values("importance", ascending=False)

    return BacktestResult(preds=preds, truth=truth, mae=mae, rmse=rmse, dir_acc=dir_acc, importance=importance)


def train_latest_model(df: pd.DataFrame):
    """
    Train on all available data and return the model + the latest row prediction.
    """
    train = df.copy()
    X = train[FEATURE_COLS]
    y = train["y_week_return"]

    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=500,
        random_state=42,
    )
    model.fit(X, y)

    latest_X = X.tail(1)
    latest_pred = float(model.predict(latest_X)[0])
    return model, latest_pred