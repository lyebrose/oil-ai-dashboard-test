# model.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.inspection import permutation_importance


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_5d", "vol_20d", "vol_60d",
    "price_over_ma20", "price_over_ma50",
    "drawdown_20d",
]


def _make_pipeline() -> Pipeline:
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


@dataclass
class BacktestResult:
    preds: pd.Series
    truth: pd.Series
    mae: float
    rmse: float
    dir_acc: float
    importance: pd.DataFrame


def walk_forward_backtest(df: pd.DataFrame, start_year: int = 2018) -> BacktestResult:
    d = df.copy()
    d["year"] = d.index.year

    preds_all = []
    truth_all = []

    last_pipe = None
    last_X_test = None
    last_y_test = None

    years = sorted(y for y in d["year"].unique() if y >= start_year)
    for y in years:
        train = d[d["year"] < y]
        test = d[d["year"] == y]
        if len(train) < 250 or len(test) < 40:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train["y_week_return"]
        X_test = test[FEATURE_COLS]
        y_test = test["y_week_return"]

        pipe = _make_pipeline()
        pipe.fit(X_train, y_train)

        pred = pd.Series(pipe.predict(X_test), index=X_test.index)
        preds_all.append(pred)
        truth_all.append(y_test)

        last_pipe = pipe
        last_X_test = X_test
        last_y_test = y_test

    preds = pd.concat(preds_all).sort_index()
    truth = pd.concat(truth_all).sort_index()

    mae = float(mean_absolute_error(truth, preds))
    rmse = float(np.sqrt(mean_squared_error(truth, preds)))
    dir_acc = float(accuracy_score((truth > 0).astype(int), (preds > 0).astype(int)))

    importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": 0.0})
    if last_pipe is not None and last_X_test is not None and len(last_X_test) > 20:
        r = permutation_importance(last_pipe, last_X_test, last_y_test, n_repeats=10, random_state=42)
        importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": r.importances_mean})
        importance = importance.sort_values("importance", ascending=False)

    return BacktestResult(preds=preds, truth=truth, mae=mae, rmse=rmse, dir_acc=dir_acc, importance=importance)


def train_latest_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["y_week_return"]

    pipe = _make_pipeline()
    pipe.fit(X, y)

    latest_pred = float(pipe.predict(X.tail(1))[0])
    return pipe, latest_pred