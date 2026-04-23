import numpy as np
from typing import Dict, List, Optional
from ..core.validators import BaseValidator
from ..core.results import Result

class ProbabilisticValidator(BaseValidator):
    def validate(self, *args, **kwargs) -> List[Result]:
        return []
    
    def validate_prediction_intervals(self, y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                                      confidence_level: float = 0.95) -> List[Result]:
        results = []
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        coverage_deviation = abs(coverage - confidence_level)
        results.append(Result(test_name="Prediction Interval Coverage",
                                        statistic=coverage,
                                        passed=coverage_deviation < 0.05,  # within 5% of target
                                        metadata={'target_coverage': confidence_level, 
                                                  'actual_coverage': coverage, 
                                                  'deviation': coverage_deviation}))
        avg_width = np.mean(y_upper - y_lower)
        std_y = np.std(y_true)
        relative_width = avg_width / std_y if std_y > 0 else np.nan
        results.append(Result(test_name="Average Interval Width",
                                        statistic=avg_width,
                                        metadata={'relative_width': relative_width,
                                                  'interpretation': 'Width relative to data std'}))
        alpha = 1 - confidence_level
        width = y_upper - y_lower
        lower_miss = (2 / alpha) * (y_lower - y_true) * (y_true < y_lower)
        upper_miss = (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
        winkler_score = np.mean(width + lower_miss + upper_miss)
        results.append(Result(test_name="Winkler Score",
                                        statistic=winkler_score,
                                        metadata={'interpretation': 'Lower is better; penalizes wide intervals and misses'}))
        sharpness = np.mean(y_upper - y_lower)
        results.append(Result(test_name="Interval Sharpness",
                                        statistic=sharpness,
                                        metadata={'interpretation': 'Average interval width; smaller indicates sharper forecasts'}))
        return results
    
    def validate_quantile_forecasts(self, y_true: np.ndarray, quantile_forecasts: Dict[float, np.ndarray]) -> List[Result]:
        results = []
        for q, y_pred_q in sorted(quantile_forecasts.items()):
            error = y_true - y_pred_q
            loss = np.where(error >= 0, q * error, (q - 1) * error)
            pinball_score = np.mean(loss)
            coverage = np.mean(y_true <= y_pred_q)
            coverage_deviation = abs(coverage - q)
            results.append(Result(test_name=f"Quantile {q:.2f} Validation",
                                            statistic=pinball_score,
                                            p_value=coverage,
                                            passed=coverage_deviation < 0.05,
                                            metadata={'target_quantile': q,
                                                      'empirical_coverage': coverage,
                                                      'coverage_deviation': coverage_deviation,
                                                      'pinball_loss': pinball_score,
                                                      'interpretation': 'Coverage should match target quantile'}))
        return results
    
    def validate_probabilistic_calibration(self, y_true: np.ndarray, predicted_probs: np.ndarray, 
                                           n_bins: int = 10) -> List[Result]:
        results = []
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        calibration_errors = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_predicted = np.mean(predicted_probs[mask])
                avg_actual = np.mean(y_true[mask])
                calibration_errors.append(abs(avg_predicted - avg_actual))
        ece = np.mean(calibration_errors) if calibration_errors else np.nan
        results.append(Result(
            test_name="Expected Calibration Error",
            statistic=ece,
            passed=ece < 0.1, # 10%
            metadata={'interpretation': 'Lower is better; measures calibration accuracy', 'n_bins': n_bins}))
        return results
    
    def validate_continuous_ranked_probability_score(self, y_true: np.ndarray, forecast_samples: np.ndarray) -> Result:
        n_samples = forecast_samples.shape[1]
        term1 = np.mean([np.mean(np.abs(forecast_samples[:, i] - y_true)) for i in range(n_samples)])
        term2 = 0
        count = 0
        for i in range(min(n_samples, 100)): 
            for j in range(i + 1, min(n_samples, 100)):
                term2 += np.mean(np.abs(forecast_samples[:, i] - forecast_samples[:, j]))
                count += 1
        term2 = term2 / count if count > 0 else 0
        crps = term1 - 0.5 * term2

        return Result(test_name="Continuous Ranked Probability Score", statistic=crps, 
                                metadata={'interpretation': 'Lower is better; measures forecast skill', 'n_samples': n_samples})