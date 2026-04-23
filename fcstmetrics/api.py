import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple, Union
from .core.framework import ModelValidationFramework
from .core.results import Output
from .extensions.comparison import ModelComparator
from .extensions.backtesting import Backtester, BacktestResults
ArrayLike = Union[np.ndarray, pd.Series]
# expected shape: (y_train, y_train_pred, y_test, y_test_pred)
ModelDataTuple = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]

def quick_validate(y_true: ArrayLike, y_pred: ArrayLike, model_name: str = "Model", print_report: bool = True) -> Output:
    framework = ModelValidationFramework(model_name=model_name)
    framework.calculate_metrics(y_true, y_pred, prefix="")
    framework.validate_residuals(framework._coerce_1d(y_true, "y_true") - framework._coerce_1d(y_pred, "y_pred"))
    if print_report:
        framework.print_summary()
    return framework.get_report()

def validate_forecast(y_train: ArrayLike, y_train_pred: ArrayLike, y_test: ArrayLike, y_test_pred: ArrayLike, model_name: str = "Forecast Model",
                      print_report: bool = True, check_stationarity: bool = True) -> Output:
    framework = ModelValidationFramework(model_name=model_name)
    report = framework.run_comprehensive_validation(y_train=y_train, 
                                                    y_train_pred=y_train_pred, 
                                                    y_test=y_test, 
                                                    y_test_pred=y_test_pred, 
                                                    check_stationarity=check_stationarity)
    if print_report:
        framework.print_summary()
    return report

def compare_models(models_data: Dict[str, ModelDataTuple], print_report: bool = True) -> ModelComparator:
    if not models_data:
        raise ValueError("models_data must contain at least one entry")
    comparator = ModelComparator()
    for name, data in models_data.items():
        if len(data) != 4:
            raise ValueError(f"Entry for '{name}' must be a 4-tuple (y_train, y_train_pred, y_test, y_test_pred); got length {len(data)}")
        y_train, y_train_pred, y_test, y_test_pred = data
        comparator.add_model(name, 
                             y_train, 
                             y_train_pred, 
                             y_test, 
                             y_test_pred)
    if print_report:
        print(comparator.generate_report())
    return comparator

def run_backtest(data: pd.Series, all_predictions: np.ndarray, initial_window: int, step_size: int = 1, forecasting_horizon: int = 1,
                 model_name: str = "Backtest", raise_on_error: bool = False) -> BacktestResults:
    backtester = BacktestingFramework(initial_window=initial_window,
                                      step_size=step_size,
                                      forecasting_horizon=forecasting_horizon,
                                      raise_on_error=raise_on_error)
    return backtester.backtest_with_predictions(data, all_predictions, model_name)

def run_backtest_with_model(data: pd.Series, model_factory: Callable, initial_window: int, step_size: int = 1, forecasting_horizon: int = 1,
                            refit_frequency: int = 1, fit_params: Optional[Dict] = None, predict_params: Optional[Dict] = None,
                            model_name: str = "Backtest", raise_on_error: bool = False) -> BacktestResults:
    backtester = BacktestingFramework(initial_window=initial_window,
                                      step_size=step_size,
                                      forecasting_horizon=forecasting_horizon,
                                      refit_frequency=refit_frequency,
                                      raise_on_error=raise_on_error)
    return backtester.backtest_with_model(data=data,
                                          model_factory=model_factory,
                                          fit_params=fit_params,
                                          predict_params=predict_params,
                                          model_name=model_name)

