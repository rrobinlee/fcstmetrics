import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats

def plot_eda(series: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x_axis = timestamps if timestamps is not None else np.arange(len(series))
  
    axes[0, 0].plot(x_axis, series, linewidth=2, color='#2C3E50')
    axes[0, 0].fill_between(x_axis, series, alpha=0.3, color='#3498DB')
    axes[0, 0].set_title(f'{series.name}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date', fontsize=11)
    axes[0, 0].set_ylabel(f'{series.name}', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(series, bins=30, edgecolor='black', alpha=0.7, color='#3498DB')
    axes[0, 1].axvline(series.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_title(f'Distribution of {series.name}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel(f'{series.name}', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    yoy_change = series.diff(12)
    axes[1, 0].plot(x_axis, yoy_change, linewidth=2, color='#E74C3C')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].fill_between(x_axis, yoy_change, 0, 
                             where=(yoy_change > 0), alpha=0.3, color='green', label='Increase')
    axes[1, 0].fill_between(x_axis, yoy_change, 0, 
                             where=(yoy_change <= 0), alpha=0.3, color='red', label='Decrease')
    axes[1, 0].set_title('Year-over-Year Change', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Date', fontsize=11)
    axes[1, 0].set_ylabel('YoY Change', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
    axes[1, 1].plot(x_axis, series, linewidth=1, alpha=0.5, label='Original', color='gray')
    axes[1, 1].plot(x_axis, rolling_mean, linewidth=2, label='12-Month MA', color='#3498DB')
    axes[1, 1].fill_between(x_axis, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std,
                             alpha=0.2, color='#3498DB', label='+/- Std Dev')
    axes[1, 1].set_title('Rolling Statistics (12-Month Window)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date', fontsize=11)
    axes[1, 1].set_ylabel(f'{series.name}', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_residuals(residuals: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None, figsize: Tuple[int, int] = (14, 10),
                   title: str = "Residual Analysis") -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x_axis = timestamps if timestamps is not None else np.arange(len(residuals))
    axes[0, 0].plot(x_axis, residuals, linewidth=1.5, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time' if timestamps is not None else 'Index')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30,
                    'r-', linewidth=2, label='Normal')
    axes[0, 1].legend()
    
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(np.arange(len(residuals)), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Residuals vs Index', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None, train_test_split: Optional[int] = None, 
                     figsize: Tuple[int, int] = (14, 6), title: str = "Predictions vs Actuals") -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    x_axis = timestamps if timestamps is not None else np.arange(len(y_true))
    ax.plot(x_axis, y_true, linewidth=2, label='Actual', alpha=0.8)
    ax.plot(x_axis, y_pred, linewidth=2, label='Predicted', alpha=0.8)
    if train_test_split is not None:
        split_x = x_axis[train_test_split] if timestamps is not None else train_test_split
        ax.axvline(x=split_x, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time' if timestamps is not None else 'Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_diagnostics(y_train: np.ndarray, y_train_pred: np.ndarray, y_test: np.ndarray, y_test_pred: np.ndarray,
                     figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y_train, linewidth=2, label='Actual', alpha=0.8)
    ax1.plot(y_train_pred, linewidth=2, label='Predicted', alpha=0.8)
    ax1.set_title('Training Set', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(y_test, linewidth=2, label='Actual', alpha=0.8)
    ax2.plot(y_test_pred, linewidth=2, label='Predicted', alpha=0.8)
    ax2.set_title('Test Set', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    train_residuals = y_train - y_train_pred
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(train_residuals, linewidth=1.5)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title('Training Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    test_residuals = y_test - y_test_pred
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(test_residuals, linewidth=1.5, color='orange')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_title('Test Residuals', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(train_residuals, bins=30, alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--')
    ax5.set_title('Training Residual Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Residual')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(test_residuals, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax6.axvline(x=0, color='red', linestyle='--')
    ax6.set_title('Test Residual Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Residual')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Diagnostics', fontsize=16, fontweight='bold')
    return fig
