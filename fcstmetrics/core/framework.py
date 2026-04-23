import sys
import numpy as np
from typing import Optional, IO
from .validators import ResidualValidator, StationarityValidator
from .metrics import MetricsCalculator
from .results import Output


class ModelValidationFramework:
    def __init__(self, model_name: str = "Model", alpha: float = 0.05, max_lags: int = 10, mape_warn_threshold: float = 15.0, r2_warn_threshold: float = 0.3):
        self.model_name = model_name
        self.alpha = alpha
        self.max_lags = max_lags
        self.mape_warn_threshold = mape_warn_threshold
        self.r2_warn_threshold = r2_warn_threshold
        self._residual_validator = ResidualValidator(alpha, max_lags)
        self._stationarity_validator = StationarityValidator(alpha)
        self._metrics_calculator = MetricsCalculator()
        self.report = Output(model_name=model_name)

    def reset(self) -> "ModelValidationFramework":
        self.report = Output(model_name=self.model_name)
        return self

    def validate_residuals(self, residuals: np.ndarray) -> "ModelValidationFramework":
        residuals = self._coerce_1d(residuals, "residuals")
        for result in self._residual_validator.validate(residuals):
            self.report.add_result(result)
        self._generate_test_warnings()
        return self

    def validate_stationarity(self, series: np.ndarray) -> "ModelValidationFramework":
        series = self._coerce_1d(series, "series")
        for result in self._stationarity_validator.validate(series):
            self.report.add_result(result)
        self._generate_test_warnings()
        return self

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> "ModelValidationFramework":
        y_true, y_pred = self._coerce_matched_1d(y_true, y_pred, prefix or "arrays")
        metrics = self._metrics_calculator.calculate_all_metrics(y_true, y_pred, prefix)
        for name, value in metrics.items():
            self.report.add_metric(name, value)
        self._generate_metric_warnings()
        return self

    def run_comprehensive_validation(self, y_train: np.ndarray, y_train_pred: np.ndarray, y_test: Optional[np.ndarray] = None, 
                                     y_test_pred: Optional[np.ndarray] = None, check_stationarity: bool = True) -> Output:
        self.reset()
        self.calculate_metrics(y_train, y_train_pred, prefix="train_")
        if (y_test is None) != (y_test_pred is None):
            raise ValueError("Provide both y_test and y_test_pred, or neither.")
        if y_test is not None:
            self.calculate_metrics(y_test, y_test_pred, prefix="test_")
        train_residuals = (self._coerce_1d(y_train, "y_train") - self._coerce_1d(y_train_pred, "y_train_pred"))
        self.validate_residuals(train_residuals)
        if check_stationarity:
            self.validate_stationarity(y_train)
        return self.report

    def get_report(self) -> Output:
        return self.report

    def to_dict(self) -> dict:
        return {"model_name": self.report.model_name,
                "timestamp": str(self.report.timestamp),
                "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in self.report.metrics.items()},
                "tests": [{"name": r.test_name,
                           "passed": r.passed,
                           "p_value": (float(r.p_value) if r.p_value is not None else None)} 
                          for r in self.report.results],
                "warnings": list(self.report.warnings)}

    def print_summary(self, file: IO = sys.stdout) -> None:
        sep = "=" * 80
        thin = "-" * 80
        def p(line: str = "") -> None:
            print(line, file=file)
        p(sep)
        p(f"Evaluation Summary: {self.model_name}")
        p(sep)
        p(f"Timestamp: {self.report.timestamp}")
        p()
        if self.report.metrics:
            p("Metrics:")
            p(thin)
            for name, value in sorted(self.report.metrics.items()):
                if isinstance(value, float) and not np.isnan(value):
                    p(f"  {name:30s}: {value:12.4f}")
                else:
                    p(f"  {name:30s}: {value}")
            p()
        if self.report.results:
            p("Tests:")
            p(thin)
            for result in self.report.results:
                p(f"  {result}")
            p()
        if self.report.warnings:
            p("Warnings:")
            p(thin)
            for warning in self.report.warnings:
                p(f"  {warning}")
            p()

        p(sep)

    @staticmethod
    def _coerce_1d(arr: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).flatten()
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"'{name}' must be a non-empty 1-D array.")
        return arr

    def _coerce_matched_1d(self, y_true: np.ndarray, y_pred: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        y_true = self._coerce_1d(y_true, f"{label} y_true")
        y_pred = self._coerce_1d(y_pred, f"{label} y_pred")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch for '{label}': "
                f"y_true={y_true.shape}, y_pred={y_pred.shape}."
            )
        return y_true, y_pred

    def _generate_test_warnings(self) -> None:
        for result in self.report.results:
            if result.passed is False:
                msg = f"Failed: {result.test_name}"
                if result.p_value is not None:
                    msg += f" (p={result.p_value:.4f})"
                if msg not in self.report.warnings:
                    self.report.add_warning(msg)

    def _generate_metric_warnings(self) -> None:
        mape = self.report.metrics.get("test_MAPE")
        if mape is not None and mape > self.mape_warn_threshold:
            msg = f"High MAPE on test set: {mape:.2f}% (threshold: {self.mape_warn_threshold}%)"
            if msg not in self.report.warnings:
                self.report.add_warning(msg)

        r2 = self.report.metrics.get("test_R2")
        if r2 is not None and r2 < self.r2_warn_threshold:
            msg = f"Low R2 on test set: {r2:.4f} (threshold: {self.r2_warn_threshold})"
            if msg not in self.report.warnings:
                self.report.add_warning(msg)