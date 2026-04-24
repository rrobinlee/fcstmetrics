import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict

def convert_to_sktime_format(data: Union[pd.DataFrame, pd.Series, np.ndarray], freq: Optional[str] = None,
                             start_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError(f"DataFrame has {len(data.columns)} columns\nPlease select a single column or pass a Series/array")
    if isinstance(data, np.ndarray):
        data = pd.Series(data.flatten())
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    if isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if data.name is None:
            data.name = 'value'
        return data
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()
    else:
        start_date = pd.Timestamp(start_date)
    if freq is None:
        freq = 'D'
        warnings.warn(f"No frequency provided. Defaulting to daily frequency ('{freq}')\nSpecify freq parameter for accurate time indexing",
                      UserWarning)
    try:
        new_index = pd.date_range(start=start_date, periods=len(data), freq=freq)
        data.index = new_index
    except Exception as e:
        raise ValueError(f"Could not create datetime index with freq='{freq}': {e}")
    if data.name is None:
        data.name = 'value'
    return data

def prepare_sktime_train_test(data: Union[pd.Series, np.ndarray], train_size: Optional[Union[int, float]] = None,
                              test_size: Optional[Union[int, float]] = None, freq: Optional[str] = None,
                              start_date: Optional[Union[str, pd.Timestamp]] = None, include_plot = False) -> Tuple[pd.Series, pd.Series]:
    series = convert_to_sktime_format(data, freq=freq, start_date=start_date)
    n = len(series)
    if train_size is not None:
        if isinstance(train_size, float) and 0 < train_size < 1:
            split_idx = int(n * train_size)
        elif isinstance(train_size, int):
            split_idx = train_size
        else:
            raise ValueError("train_size must be float in (0,1) or positive integer")
    elif test_size is not None:
        if isinstance(test_size, float) and 0 < test_size < 1:
            split_idx = int(n * (1 - test_size))
        elif isinstance(test_size, int):
            split_idx = n - test_size
        else:
            raise ValueError("test_size must be float in (0,1) or positive integer")
    else:
        split_idx = int(n * 0.8)
    y_train = series.iloc[:split_idx]
    y_test = series.iloc[split_idx:]
    if include_plot:
        plt.figure(figsize=(14, 6))
        plt.plot(y_train.index, y_train, linewidth=2, label='Training Data', color='#3498DB')
        plt.plot(y_test.index, y_test, linewidth=2, label='Test Data', color='#E74C3C')
        plt.axvline(x=y_test.index[0], color='black', linestyle='--', linewidth=2, alpha=0.7, label='Split Point')
        plt.title('Train/Test Split', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=11)
        plt.ylabel(f'{data.name}', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return y_train, y_test

def create_sktime_dataset_from_arrays(y_train: np.ndarray, y_test: np.ndarray, y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                                      freq: str = 'D', start_date: Optional[Union[str, pd.Timestamp]] = None) -> Dict[str, pd.Series]:
    n_train = len(y_train)
    n_total = n_train + len(y_test)
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()
    full_index = pd.date_range(start=start_date,
                               periods=n_total,
                               freq=freq)
    y_train_series = pd.Series(y_train, 
                               index=full_index[:n_train],
                               name='actual')
    y_test_series = pd.Series(y_test, 
                              index=full_index[n_train:], 
                              name='actual')
    y_train_pred_series = pd.Series(y_train_pred, 
                                    index=full_index[:n_train],
                                    name='predicted')
    y_test_pred_series = pd.Series(y_test_pred, 
                                   index=full_index[n_train:],
                                   name='predicted')
    return {'y_train': y_train_series, 
            'y_test': y_test_series, 
            'y_train_pred': y_train_pred_series,
            'y_test_pred': y_test_pred_series}
