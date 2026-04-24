## Reference

### Residual

| Test | Null Hypothesis | Purpose |
|------|----------------|---------|
| **Jarque-Bera** | Residuals are normally distributed | Test normality |
| **Shapiro-Wilk** | Residuals are normally distributed | Test normality (small samples) |
| **Kolmogorov-Smirnov** | Residuals follow normal distribution | Test normality |
| **Ljung-Box** | No autocorrelation in residuals | Detect autocorrelation |
| **Durbin-Watson** | No first-order autocorrelation | Quick autocorrelation check |
| **Zero Mean Test** | Mean of residuals = 0 | Check for bias |
| **Levene's Test** | Constant variance over time | Test homoscedasticity |

### Stationarity

| Test | Null Hypothesis | Interpretation |
|------|----------------|----------------|
| **ADF (Augmented Dickey-Fuller)** | Series has unit root (non-stationary) | Reject H0 → Stationary |
| **KPSS** | Series is stationary | Fail to reject H0 → Stationary |

### Performance

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MSE** | Mean Squared Error | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 10% excellent, < 20% good |
| **R2** | Coefficient of Determination | > 0.7 good, > 0.9 excellent |
| **NRMSE** | Normalized RMSE | < 0.5 acceptable |
| **Directional Accuracy** | % of correct direction predictions | > 50% better than random |
| **Bias** | Mean residual | Close to 0 |

### Common Issues and Solutions

| Issue | Symptom | Potential Solutions |
|-------|---------|-------------------|
| **Non-normal residuals** | Failed normality tests | Transform target variable, check for outliers |
| **Autocorrelated residuals** | Failed Ljung-Box | Add AR terms, increase model order |
| **Heteroscedasticity** | Failed Levene's test | Log transform, use weighted regression |
| **High MAPE** | MAPE > 20% | Feature engineering, try different model |
| **Low R2** | R2 < 0.5 | Add features, increase model complexity |
| **Non-stationary series** | Failed ADF/KPSS | Difference the series, add trend terms |

## Examples

### Evaluation

```python
# basic validation
from fcstmetrics import quick_validate
report = quick_validate(y_test, predictions, model_name="model name")

# comprehensive validation
from fcstmetrics import validate_forecast
report = validate_forecast(y_train, train_preds, y_test, test_preds, model_name="SARIMA")
report.summary() # summary dataframe

# comparing multiple models
from fcstmetrics import compare_models
models = {'ARIMA': (y_train, arima_train, y_test, arima_test),
          'Prophet': (y_train, prophet_train, y_test, prophet_test),
          'LSTM': (y_train, lstm_train, y_test, lstm_test)}
comparator = compare_models(models)
comparison_df = comparator.compare_metrics()
dm_result = comparator.diebold_mariano_test('ARIMA', 'Prophet')

# cross-validation
from fcstmetrics.extensions import TimeSeriesCrossValidator, cross_validate_model
cv = TimeSeriesCrossValidator(n_splits=5, strategy='rolling', gap=0)
cv_results = cross_validate_model(y, y_pred_all, cv, model_name="model name")
summary_df = cv_results.summary()
print(cv_results.aggregate_metrics)

# backtesting
from fcstmetrics import run_backtest
results = run_backtest(data=time_series, all_predictions=predictions, initial_window=100, step_size=1, forecasting_horizon=10)
print(results.validation_report.metrics)
df = results.to_dataframe()

# probabilistic intervals
from fcstmetrics.extensions import ProbabilisticValidator
prob_validator = ProbabilisticValidator()
interval_results = prob_validator.validate_prediction_intervals(y_test, lower_bounds, upper_bounds, confidence_level=0.95)
quantile_forecasts = {0.1: q10_predictions, 0.5: q50_predictions, 0.9: q90_predictions}
quantile_results = prob_validator.validate_quantile_forecasts( y_test, quantile_forecasts)
```

### Visualization
```python
from fcstmetrics.utils import plot_residuals, plot_predictions, plot_diagnostics
fig = plot_residuals(residuals, timestamps=data.index)
fig = plot_predictions(y_test, predictions, timestamps=test_dates)
fig = plot_diagnostics(y_train, train_pred, y_test, test_pred)
```

### Other
```python
# custom validator
from fcstmetrics.core import BaseValidator, ValidationResult
class MyCustomValidator(BaseValidator):
    def validate(self, data):
        result = ValidationResult(test_name="My Custom Test", statistic=some_value, 
                                  p_value=p_val, passed=p_val > self.alpha)
        return [result]
from fcstmetrics.core import ModelValidationFramework
framework = ModelValidationFramework()
framework.report.results.extend(MyCustomValidator().validate(data))

# advanced residual validation
from fcstmetrics.core import AdvancedResidualValidator
validator = AdvancedResidualValidator(alpha=0.05, max_lags=20)
results = validator.validate(residuals)
# runs test for randomness and outlier detection