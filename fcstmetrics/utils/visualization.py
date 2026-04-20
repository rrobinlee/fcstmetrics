import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats

def plot_eda(series: pd.Series, timestamps: Optional[pd.DatetimeIndex] = None) -> plt.Figure:
    series_name = getattr(series, 'name', None) or 'Value'
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.close(fig)
    
    x_axis = timestamps if timestamps is not None else np.arange(len(series))
    series_vals = series.values if hasattr(series, 'values') else series

    axes[0, 0].plot(x_axis, series_vals, linewidth=2, color='#2C3E50')
    axes[0, 0].fill_between(x_axis, series_vals, alpha=0.3, color='#3498DB')
    axes[0, 0].set_title(series_name, fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date', fontsize=11)
    axes[0, 0].set_ylabel(series_name, fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(series_vals, bins=30, edgecolor='black', alpha=0.7, color='#3498DB')
    axes[0, 1].axvline(series_vals.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_title(f'Distribution of {series_name}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel(series_name, fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if hasattr(series, 'diff'):
        yoy_change = series.diff(12)
        valid = yoy_change.notna()
        x_valid = x_axis[valid] if hasattr(x_axis, '__getitem__') else np.array(x_axis)[valid]
        yoy_valid = yoy_change[valid].values
    else:
        yoy_change = np.diff(series_vals, n=1, prepend=[np.nan] * 12)[:len(series_vals)]
        valid = ~np.isnan(yoy_change)
        x_valid = np.array(x_axis)[valid]
        yoy_valid = yoy_change[valid]

    axes[1, 0].plot(x_axis, yoy_change if not hasattr(yoy_change, 'values') else yoy_change.values,
                    linewidth=2, color='#E74C3C')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].fill_between(x_valid, yoy_valid, 0,
                             where=(yoy_valid > 0), alpha=0.3, color='green', label='Increase')
    axes[1, 0].fill_between(x_valid, yoy_valid, 0,
                             where=(yoy_valid <= 0), alpha=0.3, color='red', label='Decrease')
    axes[1, 0].set_title('Year-over-Year Change', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Date', fontsize=11)
    axes[1, 0].set_ylabel('YoY Change', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if hasattr(series, 'rolling'):
        rolling_mean = series.rolling(window=12).mean().values
        rolling_std = series.rolling(window=12).std().values
    else:
        rolling_mean = np.convolve(series_vals, np.ones(12) / 12, mode='full')[:len(series_vals)]
        rolling_mean[:11] = np.nan
        rolling_std = np.array([series_vals[max(0, i-11):i+1].std() for i in range(len(series_vals))])
        rolling_std[:11] = np.nan

    axes[1, 1].plot(x_axis, series_vals, linewidth=1, alpha=0.5, label='Original', color='gray')
    axes[1, 1].plot(x_axis, rolling_mean, linewidth=2, label='12-Month MA', color='#3498DB')
    axes[1, 1].fill_between(x_axis, rolling_mean - 2 * rolling_std, rolling_mean + 2 * rolling_std,
                             alpha=0.2, color='#3498DB', label='+/- 2 Std Dev')
    axes[1, 1].set_title('Rolling Statistics (12-Month Window)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date', fontsize=11)
    axes[1, 1].set_ylabel(series_name, fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_residuals(residuals: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None,
                   figsize: Tuple[int, int] = (14, 10),
                   title: str = "Residual Analysis") -> plt.Figure:
    residuals = np.asarray(residuals)  
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.close(fig)  
    x_axis = timestamps if timestamps is not None else np.arange(len(residuals))

    axes[0, 0].plot(x_axis, residuals, linewidth=1.5, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time' if timestamps is not None else 'Index')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    mu, sigma = residuals.mean(), residuals.std()
    n, bin_edges, _ = axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    bin_width = bin_edges[1] - bin_edges[0] 
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * bin_width,
                    'r-', linewidth=2, label='Normal fit')
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(np.arange(len(residuals)), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Residuals vs Index', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None,
                     train_test_split: Optional[int] = None,
                     figsize: Tuple[int, int] = (14, 6), title: str = "Predictions vs Actuals") -> plt.Figure:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}")
    if train_test_split is not None and not (0 < train_test_split < len(y_true)):
        raise ValueError(f"train_test_split ({train_test_split}) must be between 0 and {len(y_true)}")
    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)
    x_axis = timestamps if timestamps is not None else np.arange(len(y_true))
    ax.plot(x_axis, y_true, linewidth=2, label='Actual', alpha=0.8)
    ax.plot(x_axis, y_pred, linewidth=2, label='Predicted', alpha=0.8, linestyle='--')
    if train_test_split is not None:
        split_x = x_axis[train_test_split]
        ax.axvline(x=split_x, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time' if timestamps is not None else 'Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_diagnostics(y_train: np.ndarray, y_train_pred: np.ndarray,
                     y_test: np.ndarray, y_test_pred: np.ndarray,
                     figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
    y_train = np.asarray(y_train)
    y_train_pred = np.asarray(y_train_pred)
    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)
    if len(y_train) != len(y_train_pred):
        raise ValueError(f"y_train and y_train_pred must have the same length, got {len(y_train)} and {len(y_train_pred)}")
    if len(y_test) != len(y_test_pred):
        raise ValueError(f"y_test and y_test_pred must have the same length, got {len(y_test)} and {len(y_test_pred)}")

    fig = plt.figure(figsize=figsize)
    plt.close(fig)

    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    for ax, y_true, y_pred, label in [(fig.add_subplot(gs[0, 0]), y_train, y_train_pred, 'Training Set'),
                                      (fig.add_subplot(gs[0, 1]), y_test,  y_test_pred,  'Test Set')]:
        ax.plot(y_true, linewidth=2, label='Actual', alpha=0.8)
        ax.plot(y_pred, linewidth=2, label='Predicted', alpha=0.8, linestyle='--')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    residual_ylim = (min(train_residuals.min(), test_residuals.min()),
                     max(train_residuals.max(), test_residuals.max()))
    for ax, residuals, label in [(fig.add_subplot(gs[1, 0]), train_residuals, 'Training Residuals'),
                                 (fig.add_subplot(gs[1, 1]), test_residuals,  'Test Residuals')]:
        ax.plot(residuals, linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Residual')
        ax.set_ylim(residual_ylim)
        ax.grid(True, alpha=0.3)

    shared_bins = np.linspace(residual_ylim[0], residual_ylim[1], 31) 
    for ax, residuals, label in [(fig.add_subplot(gs[2, 0]), train_residuals, 'Training Residual Distribution'),
                                 (fig.add_subplot(gs[2, 1]), test_residuals,  'Test Residual Distribution')]:
        ax.hist(residuals, bins=shared_bins, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Model Diagnostics', fontsize=16, fontweight='bold')
    return fig
