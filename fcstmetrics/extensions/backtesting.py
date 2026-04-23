import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..core.results import Output
from ..core.framework import ModelValidationFramework
logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    predictions: List[np.ndarray]
    actuals: List[np.ndarray]
    timestamps: List
    validation_report: Output
    metadata: Dict[str, Any]
    fold_errors: List[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.predictions) != len(self.actuals):
            raise ValueError(f"predictions ({len(self.predictions)}) and actuals ({len(self.actuals)}) must have the same length")
        self.fold_errors = [a - p for a, p in zip(self.actuals, self.predictions)]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.predictions:
            return pd.DataFrame(columns=["timestamp", 
                                         "actual", 
                                         "predicted", 
                                         "error", 
                                         "abs_error", 
                                         "squared_error", 
                                         "fold"])
        fold_ids, timestamps = [], []
        for fold_idx, (ts, pred) in enumerate(zip(self.timestamps, self.predictions)):
            n = len(pred)
            fold_ids.extend([fold_idx] * n)
            timestamps.extend([ts] * n)
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.actuals)
        errors = y_true - y_pred
        return pd.DataFrame({"fold": fold_ids,
                             "timestamp": timestamps,
                             "actual": y_true, 
                             "predicted": y_pred,
                             "error": errors, 
                             "abs_error": np.abs(errors), 
                             "squared_error": errors ** 2})

    def summary_stats(self) -> Dict[str, float]:
        df = self.to_dataframe()
        if df.empty:
            return {}
        return {"n_folds": len(self.predictions), 
                "n_observations": len(df), 
                "mean_error": float(df["error"].mean()), 
                "std_error": float(df["error"].std()),
                "mean_abs_error": float(df["abs_error"].mean()), 
                "rmse": float(np.sqrt(df["squared_error"].mean())),
                "min_error": float(df["error"].min()),
                "max_error": float(df["error"].max()), 
                "pct_positive_error": float((df["error"] > 0).mean() * 100)}

    def per_fold_stats(self) -> pd.DataFrame:
        rows = []
        for fold_idx, (ts, errors) in enumerate(zip(self.timestamps, self.fold_errors)):
            rows.append({"fold": fold_idx, "timestamp": ts, "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                         "bias": float(np.mean(errors)), "n": len(errors)})
        return pd.DataFrame(rows)

    def rolling_errors(self, window: int = 10) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.assign(rolling_mae=df["abs_error"].rolling(window).mean(), 
                         rolling_rmse=np.sqrt(df["squared_error"].rolling(window).mean()),
                         rolling_bias=df["error"].rolling(window).mean(), 
                         rolling_std=df["error"].rolling(window).std())[["fold", "timestamp", "rolling_mae", "rolling_rmse","rolling_bias", "rolling_std"]]

def _build_report(predictions: List[np.ndarray], actuals: List[np.ndarray], model_name: str) -> Output:
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(actuals)
    framework = ModelValidationFramework(model_name)
    framework.calculate_metrics(y_true, y_pred, prefix = "backtest_")
    framework.validate_residuals(y_true - y_pred)
    return framework.get_report()

def _extract_forecast(forecast_raw: Any, horizon: int) -> Optional[np.ndarray]:
    if hasattr(forecast_raw, "values"):
        arr = forecast_raw.values
    elif hasattr(forecast_raw, "__array__"):
        arr = np.asarray(forecast_raw)
    elif isinstance(forecast_raw, (list, tuple)):
        arr = np.array(forecast_raw)
    else:
        arr = np.atleast_1d(forecast_raw)
    arr = arr.flatten()
    if len(arr) != horizon:
        logger.warning("Forecast length %d != expected horizon %d; fold skipped", len(arr), horizon)
        return None
    return arr.astype(float)

# backtest_with_predictions - evaluate pre-computed forecasts
# backtest_with_model - predict inside the loop
class BacktestingFramework:
    def __init__(self, initial_window: int, step_size: int = 1, forecasting_horizon: int = 1, refit_frequency: int = 1, raise_on_error: bool = False) -> None:
        if initial_window < 1:
            raise ValueError("initial_window must be >= 1.)
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        if forecasting_horizon < 1:
            raise ValueError("forecasting_horizon must be >= 1")
        self.initial_window = initial_window
        self.step_size = step_size
        self.forecasting_horizon = forecasting_horizon
        self.refit_frequency = refit_frequency
        self.raise_on_error = raise_on_error

    def backtest_with_predictions(self, data: pd.Series, all_predictions: np.ndarray, model_name: str = "Backtest") -> BacktestResults:
        all_predictions = np.asarray(all_predictions, dtype=float).flatten()
        if len(all_predictions) != len(data):
            raise ValueError(f"all_predictions length ({len(all_predictions)}) must match data length ({len(data)})")
        predictions, actuals, timestamps = self._slice_folds(data, all_predictions)
        if not predictions:
            raise RuntimeError("No backtest folds were generated - check that initial_window and forecasting_horizon are compatible with the data length")
        report = _build_report(predictions, actuals, model_name)
        return BacktestResults(predictions=predictions, 
                               actuals=actuals, 
                               timestamps=timestamps, 
                               validation_report=report, 
                               metadata=self._base_metadata(model_name, len(predictions)))
        
    def backtest_with_model(self, data: pd.Series, model_factory: Callable, fit_params: Optional[Dict] = None, 
                            predict_params: Optional[Dict] = None, model_name: str = "Backtest") -> BacktestResults:
        fit_params = fit_params or {}
        predict_params = predict_params or {}
        n = len(data)
        predictions: List[np.ndarray] = []
        actuals: List[np.ndarray] = []
        timestamps: List = []
        n_errors = 0
        model = None
        fold_counter = 0
        origins = range(self.initial_window, n - self.forecasting_horizon, self.step_size)

        for i in origins:
            should_refit = (model is None or (self.refit_frequency > 0 and fold_counter % self.refit_frequency == 0))
            if should_refit:
                train_data = data.iloc[:i]
                model = model_factory()
                try:
                    model.fit(train_data, **fit_params)
                except Exception as exc:
                    msg = f"model.fit() failed at origin {i}: {exc}"
                    if self.raise_on_error:
                        raise RuntimeError(msg) from exc
                    logger.warning(msg)
                    n_errors += 1
                    fold_counter += 1
                    continue
            try:
                raw_forecast = model.predict(self.forecasting_horizon, **predict_params)
            except Exception as exc:
                msg = f"model.predict() failed at origin {i}: {exc}"
                if self.raise_on_error:
                    raise RuntimeError(msg) from exc
                logger.warning(msg)
                n_errors += 1
                fold_counter += 1
                continue
            forecast = _extract_forecast(raw_forecast, self.forecasting_horizon)
            if forecast is None:
                n_errors += 1
                fold_counter += 1
                continue
            actual = data.iloc[i : i + self.forecasting_horizon].values.astype(float)
            predictions.append(forecast)
            actuals.append(actual)
            timestamps.append(data.index[i])
            fold_counter += 1
        if n_errors:
            warnings.warn(
                f"{n_errors} fold(s) were skipped due to errors (enable debug logging)", RuntimeWarning, stacklevel=2)
        if not predictions:
            raise RuntimeError("All backtest folds failed - check model_factory, data, and window/horizon settings")
        report = _build_report(predictions, actuals, model_name)
        meta = self._base_metadata(model_name, len(predictions))
        meta.update({"refit_frequency": self.refit_frequency, "n_errors": n_errors})
        return BacktestResults(predictions=predictions,
                               actuals=actuals,
                               timestamps=timestamps, 
                               validation_report=report, 
                               metadata=meta)

    def rolling_window_forecast_errors(self, results: BacktestResults, window_size: int = 10) -> pd.DataFrame:
        return results.rolling_errors(window=window_size)

    def _slice_folds(self, data: pd.Series, all_predictions: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
        n = len(data)
        predictions, actuals, timestamps = [], [], []
        for i in range(self.initial_window, n - self.forecasting_horizon, self.step_size):
            end = i + self.forecasting_horizon
            predictions.append(all_predictions[i:end])
            actuals.append(data.iloc[i:end].values.astype(float))
            timestamps.append(data.index[i])
        return predictions, actuals, timestamps

    def _base_metadata(self, model_name: str, n_folds: int) -> Dict[str, Any]:
        return {"model_name": model_name,
                "initial_window": self.initial_window, 
                "step_size": self.step_size,
                "forecasting_horizon": self.forecasting_horizon,
                "n_folds": n_folds}

