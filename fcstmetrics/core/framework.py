import numpy as np
from typing import Optional
from .validators import ResidualValidator, StationarityValidator
from .metrics import MetricsCalculator
from .results import ValidationReport

class ModelValidationFramework:
    def __init__(self, model_name: str = "Model", alpha: float = 0.05, max_lags: int = 10):
        self.model_name = model_name
        self.alpha = alpha
        self.max_lags = max_lags
        self.residual_validator = ResidualValidator(alpha, max_lags)
        self.stationarity_validator = StationarityValidator(alpha)
        self.metrics_calculator = MetricsCalculator()
        self.report = ValidationReport(model_name=model_name)
        
    def validate_residuals(self, residuals: np.ndarray) -> 'ModelValidationFramework':
        results = self.residual_validator.validate(residuals)
        for result in results:
            self.report.add_result(result)
        return self
        
    def validate_stationarity(self, series: np.ndarray) -> 'ModelValidationFramework':
        results = self.stationarity_validator.validate(series)
        for result in results:
            self.report.add_result(result)
        return self
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> 'ModelValidationFramework':
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, prefix)
        for name, value in metrics.items():
            self.report.add_metric(name, value)
        return self
        
    def run_comprehensive_validation(self, y_train: np.ndarray, y_train_pred: np.ndarray, y_test: Optional[np.ndarray] = None,
                                     y_test_pred: Optional[np.ndarray] = None, check_stationarity: bool = True) -> ValidationReport:
        self.calculate_metrics(y_train, y_train_pred, prefix="train_")
        if y_test is not None and y_test_pred is not None:
            self.calculate_metrics(y_test, y_test_pred, prefix="test_")
        train_residuals = np.asarray(y_train).flatten() - np.asarray(y_train_pred).flatten()
        self.validate_residuals(train_residuals)
        if check_stationarity:
            self.validate_stationarity(y_train)
        self._generate_warnings()
        return self.report
        
    def _generate_warnings(self):
        for result in self.report.results:
            if result.passed is False:
                warning = f"Failed: {result.test_name}"
                if result.p_value is not None:
                    warning += f" (p-value: {result.p_value:.4f})"
                self.report.add_warning(warning)
        if 'test_MAPE' in self.report.metrics:
            if self.report.metrics['test_MAPE'] > 10:
                self.report.add_warning(f"High MAPE on test set: {self.report.metrics['test_MAPE']:.2f}%")
        if 'test_R2' in self.report.metrics:
            if self.report.metrics['test_R2'] < 0.5:
                self.report.add_warning(f"Low R2 on test set: {self.report.metrics['test_R2']:.4f}")
                
    def get_report(self) -> ValidationReport:
        return self.report
        
    def print_summary(self):
        print("=" * 80)
        print(f"Model Validation Summary: {self.model_name}")
        print("=" * 80)
        print(f"Timestamp: {self.report.timestamp}")
        print()
        if self.report.metrics:
            print("Performance Metrics:")
            print("-" * 80)
            for name, value in sorted(self.report.metrics.items()):
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"  {name:30s}: {value:12.4f}")
                else:
                    print(f"  {name:30s}: {value}")
            print()
        if self.report.results:
            print("Diagnostic Tests:")
            print("-" * 80)
            for result in self.report.results:
                print(f"  {result}")
            print()
        if self.report.warnings:
            print("Warnings:")
            print("-" * 80)
            for warning in self.report.warnings:
                print(f"  • {warning}")
            print()
        print("=" * 80)