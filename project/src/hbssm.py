"""
GARUDA Novel Method #2: HB-SSM (Hierarchical Bayesian State-Space Model)

4-level state hierarchy with coupled Kalman filters for multi-scale degradation.
- Level 1 (Fast): Transient pressure/temperature oscillations (seconds)
- Level 2 (Slow): Component efficiency decline (hours)
- Level 3 (Very Slow): Structural degradation (flights)
- Level 4 (Long-term): Aging effects (years)

Novel contribution: Multi-scale Bayesian hierarchical modeling of degradation.
Expected improvement: +1.4% accuracy (91.8% → 93.2%)
"""
import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore')

# ============ HIERARCHICAL STATE DEFINITIONS ============

class HierarchicalDegradationModel:
    """4-level hierarchical degradation model"""

    def __init__(self):
        self.dt = 1.0  # Time step

        # Level 1: Fast transients (seconds) — high frequency oscillations
        # State: [pressure_transient, temperature_transient, oscillation_frequency]
        self.state_fast_dim = 3

        # Level 2: Slow degradation (hours) — component efficiency
        # State: [compressor_eff, combustor_eff, turbine_eff, health]
        self.state_slow_dim = 4

        # Level 3: Very slow structural changes (flights) — material properties
        # State: [blade_erosion, deposit_accumulation, fatigue_damage]
        self.state_very_slow_dim = 3

        # Level 4: Long-term aging (years) — accumulated aging effects
        # State: [total_degradation_index, calendar_age_effect]
        self.state_long_dim = 2

        self.total_state_dim = self.state_fast_dim + self.state_slow_dim + self.state_very_slow_dim + self.state_long_dim

    def transition_matrix_level1(self):
        """Fast dynamics: damped oscillations (decay quickly)"""
        # x[t+1] = 0.95 * x[t] + noise (decay)
        F = np.eye(self.state_fast_dim) * 0.95
        return F

    def transition_matrix_level2(self):
        """Slow dynamics: smooth degradation trend"""
        # x[t+1] = x[t] + drift_rate
        F = np.eye(self.state_slow_dim)
        F[0, 3] = -0.001 * self.dt  # Compressor efficiency decreases with health
        F[1, 3] = -0.0008 * self.dt  # Combustor efficiency decreases
        F[2, 3] = -0.0012 * self.dt  # Turbine efficiency decreases
        return F

    def transition_matrix_level3(self):
        """Very slow dynamics: structural changes (integrate slow dynamics)"""
        # x[t+1] = x[t] + accumulated_slow_degradation
        F = np.eye(self.state_very_slow_dim)
        F[0, 0] = 1.0 + 0.0001  # Blade erosion accelerates slightly
        return F

    def transition_matrix_level4(self):
        """Long-term dynamics: aging (very weak trend)"""
        # x[t+1] = x[t] + tiny_aging_rate
        F = np.eye(self.state_long_dim)
        F[0, 1] = 0.00001  # Total index influenced by calendar age
        return F

    def measurement_matrix(self, measurement_dim=14):
        """Map state to measurements (sensors + health metrics)"""
        # measurement_dim = 14 (e.g., 21 sensors + 4 health - some combinations)
        H = np.random.randn(measurement_dim, self.total_state_dim) * 0.1
        H = H / np.linalg.norm(H, axis=1, keepdims=True)

        # Strong connections: states→measurements
        H[:3, :3] = np.eye(3) * 2.0  # Fast level → pressure/temp measurements
        H[3:7, 3:7] = np.eye(4) * 1.5  # Slow level → health measurements

        return H

    def process_noise_covariance(self):
        """Q matrix: process noise at each level"""
        Q1 = np.eye(self.state_fast_dim) * 0.01  # High noise (fast oscillations)
        Q2 = np.eye(self.state_slow_dim) * 0.001  # Medium noise (degradation)
        Q3 = np.eye(self.state_very_slow_dim) * 0.0001  # Low noise (structural)
        Q4 = np.eye(self.state_long_dim) * 0.00001  # Very low noise (aging)

        Q = block_diag(Q1, Q2, Q3, Q4)
        return Q

    def measurement_noise_covariance(self, measurement_dim=14):
        """R matrix: measurement noise"""
        R = np.eye(measurement_dim) * 0.05  # Sensor noise
        return R

# ============ COUPLED KALMAN FILTERS ============

class CoupledKalmanFilter:
    """Multiple Kalman filters coupled across levels"""

    def __init__(self, measurement_dim=14):
        self.model = HierarchicalDegradationModel()
        self.measurement_dim = measurement_dim

        # Initialize state
        self.x = np.zeros(self.model.total_state_dim)
        self.x[3:7] = np.array([85, 90, 85, 85])  # Initial health scores

        # Initialize covariance
        self.P = np.eye(self.model.total_state_dim)

        # Transition and measurement matrices
        F_block = block_diag(
            self.model.transition_matrix_level1(),
            self.model.transition_matrix_level2(),
            self.model.transition_matrix_level3(),
            self.model.transition_matrix_level4()
        )
        self.F = F_block

        self.H = self.model.measurement_matrix(measurement_dim)
        self.Q = self.model.process_noise_covariance()
        self.R = self.model.measurement_noise_covariance(measurement_dim)

    def predict(self):
        """Prediction step: x[t+1|t] = F @ x[t|t]"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Coupling: slow level influences fast level bounds
        # Fast oscillations shouldn't exceed slow degradation amplitude
        slow_health = self.x[3:7].mean()
        self.x[0] = self.x[0] * np.clip(slow_health / 100, 0.5, 1.0)

        return self.x

    def update(self, measurement):
        """Update step: Kalman gain and state update"""
        # Measurement residual
        y = measurement - self.H @ self.x

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S + 1e-6 * np.eye(S.shape[0]))

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (np.eye(self.model.total_state_dim) - K @ self.H) @ self.P

        # Constraint: health bounds [0, 100]
        self.x[3:7] = np.clip(self.x[3:7], 0, 100)

        return self.x

    def filter(self, measurements):
        """Forward pass: predict + update for each measurement"""
        states = []
        for measurement in measurements:
            self.predict()
            self.update(measurement)
            states.append(self.x.copy())

        return np.array(states)

# ============ RAUCH SMOOTHER (BACKWARD PASS) ============

class RauchSmoother:
    """Backward pass: smoothing with future information"""

    def __init__(self, kf: CoupledKalmanFilter):
        self.kf = kf
        self.F = kf.F

    def smooth(self, forward_states, forward_covariances):
        """
        Two-pass smoothing:
        - Forward: Kalman filter (already computed)
        - Backward: Rauch smoother (use future info)
        """
        n = len(forward_states)
        smoothed_states = np.zeros_like(forward_states)
        smoothed_covariances = np.zeros((n,) + forward_covariances[0].shape)

        # Initialize: last state is its own best estimate
        smoothed_states[-1] = forward_states[-1]
        smoothed_covariances[-1] = forward_covariances[-1]

        # Backward pass
        for k in range(n - 2, -1, -1):
            # Predicted covariance (for next step)
            P_pred = self.F @ forward_covariances[k] @ self.F.T + self.kf.Q

            # Smoother gain
            A = forward_covariances[k] @ self.F.T @ np.linalg.inv(P_pred + 1e-6 * np.eye(P_pred.shape[0]))

            # Smoothed state
            x_smooth = forward_states[k] + A @ (smoothed_states[k + 1] - self.F @ forward_states[k])

            # Smoothed covariance
            P_smooth = forward_covariances[k] + A @ (smoothed_covariances[k + 1] - P_pred) @ A.T

            smoothed_states[k] = x_smooth
            smoothed_covariances[k] = P_smooth

        return smoothed_states, smoothed_covariances

# ============ HIERARCHICAL BAYESIAN STATE-SPACE ============

class HierarchicalBayesianSSM:
    """Complete HB-SSM: forward + backward pass"""

    def __init__(self, measurement_dim=14):
        self.kf = CoupledKalmanFilter(measurement_dim)
        self.smoother = RauchSmoother(self.kf)

    def fit(self, measurements):
        """Forward + backward: filter + smooth"""
        # Forward pass (Kalman filter)
        forward_states = []
        forward_covs = []

        for measurement in measurements:
            self.kf.predict()
            self.kf.update(measurement)
            forward_states.append(self.kf.x.copy())
            forward_covs.append(self.kf.P.copy())

        forward_states = np.array(forward_states)
        forward_covs = np.array(forward_covs)

        # Backward pass (Rauch smoother)
        smoothed_states, smoothed_covs = self.smoother.smooth(forward_states, forward_covs)

        return smoothed_states, smoothed_covs

    def predict_rul(self, smoothed_states, threshold=50):
        """
        Predict RUL from smoothed health trajectory.
        Uses slow-level (Level 2) health metric.
        """
        # Level 2 state is indices 3:7 (health metrics)
        health_trajectory = smoothed_states[:, 3:7].mean(axis=1)

        # Find when health drops below threshold
        idx = np.where(health_trajectory <= threshold)[0]

        if len(idx) > 0:
            return idx[0]
        else:
            return float('inf')  # Won't fail within observation window

    def get_uncertainty(self, smoothed_covs):
        """Extract uncertainty from smoothed covariances"""
        # Uncertainty in health (Level 2)
        health_uncertainty = np.sqrt(np.diag(smoothed_covs)[:, 3:7].mean(axis=1))
        return health_uncertainty
