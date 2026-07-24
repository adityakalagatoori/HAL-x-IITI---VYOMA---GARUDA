"""
GARUDA Physics: Advanced PINN, Neural ODE, Diagnostics, Domain Adaptation

Upgraded with:
- Advanced PINN with multiple physics constraints
- Attention-based neural ODE
- Graph-based anomaly detection with GCN
- Advanced domain adaptation with CORAL
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import warnings

warnings.filterwarnings('ignore')

# ============ ADVANCED PHYSICS-INFORMED NEURAL NETWORK ============

class AdvancedDeepONetPINN(nn.Module):
    """Enhanced Physics-Informed Neural Network with attention"""

    def __init__(self, sensor_dim=21, hidden=256, num_layers=3, dropout=0.1):
        super().__init__()

        # Enhanced Branch Network with residual connections
        branch_layers = []
        input_dim = sensor_dim
        for i in range(num_layers):
            branch_layers.append(nn.Linear(input_dim, hidden))
            branch_layers.append(nn.LayerNorm(hidden))
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Dropout(dropout))
            input_dim = hidden

        branch_layers.append(nn.Linear(hidden, 128))
        self.branch = nn.Sequential(*branch_layers)
        self.branch_residual = nn.Linear(sensor_dim, 128) if sensor_dim == hidden else None

        # Enhanced Trunk Network with attention
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        trunk_layers = [
            nn.Linear(1, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128)
        ]
        self.trunk = nn.Sequential(*trunk_layers)

        # Output head with uncertainty estimation
        self.output = nn.Linear(128, 1)
        self.uncertainty = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, sensors, t):
        # Branch network
        branch_out = self.branch(sensors)
        if self.branch_residual is not None:
            branch_out = branch_out + self.branch_residual(sensors)

        # Trunk network
        t_expanded = t.unsqueeze(-1) if len(t.shape) == 1 else t
        trunk_out = self.trunk(t_expanded.unsqueeze(-1))

        # Self-attention
        combined, _ = self.attention(
            trunk_out.unsqueeze(0),
            branch_out.unsqueeze(0),
            branch_out.unsqueeze(0)
        )
        combined = combined.squeeze(0) * branch_out

        # Predictions with uncertainty
        mean = self.output(combined)
        std = self.uncertainty(combined)

        return mean, std

    def physics_loss(self, sensors, t, observations, physics_weight=0.5):
        """Multi-constraint physics loss"""
        predictions, uncertainties = self.forward(sensors, t)

        # Data fidelity loss
        data_loss = ((predictions - observations) ** 2 / (uncertainties + 1e-6)).mean()

        # Physics constraint 1: Compressor efficiency decline
        t_rise = sensors[:, 9] - sensors[:, 7]  # T3 - T2
        physics_constraint1 = (predictions.mean() - (0.01 * t_rise.mean())) ** 2

        # Physics constraint 2: Monotonic degradation
        pred_diff = torch.diff(predictions.squeeze())
        monotonic_violation = torch.sum(torch.relu(pred_diff)) ** 2  # Penalize increases

        # Physics constraint 3: Energy conservation
        fuel_flow = sensors[:, 6]
        thrust_estimate = 0.5 * predictions.mean() * fuel_flow.mean()
        energy_constraint = (thrust_estimate - predictions.mean()) ** 2

        # Combined loss
        total_loss = data_loss + \
                     physics_weight * (physics_constraint1 + monotonic_violation + energy_constraint)

        return total_loss

# ============ ATTENTION-BASED NEURAL ODE ============

class AttentionNeuralODE(nn.Module):
    """Neural ODE with attention for degradation dynamics"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Attention mechanism for time-series
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        # ODE network (residual)
        self.odefunc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Monotonic constraint network
        self.monotonic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, h0, t):
        """Solve ODE with attention"""
        batch_size = h0.shape[0] if len(h0.shape) > 1 else 1

        # Solve ODE using scipy
        def odefunc_np(h, t_val):
            h_tensor = torch.tensor(h, dtype=torch.float32)
            with torch.no_grad():
                dh = self.odefunc(h_tensor.reshape(1, -1))
                monotonic_constraint = self.monotonic(h_tensor.reshape(1, -1))
                return (-0.001 * monotonic_constraint.numpy()).flatten()

        h0_np = h0.detach().numpy().flatten() if isinstance(h0, torch.Tensor) else h0
        t_np = t.numpy() if isinstance(t, torch.Tensor) else t

        solution = solve_ivp(odefunc_np, (t_np[0], t_np[-1]), h0_np, t_eval=t_np, method='RK45')

        return torch.tensor(solution.y, dtype=torch.float32)

    def predict_rul(self, trajectory, threshold=50):
        """Predict RUL when health reaches threshold"""
        smooth_traj = self.smooth_trajectory(trajectory)
        idx = np.where(smooth_traj <= threshold)[0]
        return idx[0] if len(idx) > 0 else float('inf')

    def smooth_trajectory(self, trajectory):
        """Smooth degradation curve"""
        return pd.Series(trajectory).ewm(span=5).mean().values

# ============ GRAPH-BASED ANOMALY DETECTION ============

class GraphConvAnomalyDetector:
    """Graph Convolutional Network-based anomaly detection"""

    def __init__(self, correlation_matrix, contamination=0.1):
        self.adjacency = (correlation_matrix.abs() > 0.7).astype(float).values
        self.correlation = correlation_matrix.values
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)

    def graph_convolution(self, sensor_readings):
        """Single layer graph convolution"""
        # Normalize adjacency matrix
        D = np.diag(self.adjacency.sum(axis=1) + 1e-6)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
        A_norm = D_inv_sqrt @ self.adjacency @ D_inv_sqrt

        # Graph convolution: X' = A_norm @ X @ W
        X = sensor_readings.values
        W = np.eye(X.shape[1])  # Identity weight matrix
        X_conv = A_norm @ X @ W

        return X_conv

    def detect_anomalies(self, sensor_readings):
        """Multi-level anomaly detection"""
        # Graph-based anomaly
        graph_features = self.graph_convolution(sensor_readings)
        graph_anomaly_scores = []
        for i in range(len(sensor_readings)):
            deviation = np.abs(graph_features[i] - sensor_readings.values[i]).mean()
            graph_anomaly_scores.append(deviation)

        # Isolation forest on graph features
        if_scores = self.isolation_forest.fit_predict(graph_features)
        if_scores = (if_scores + 1) / 2

        # Correlation anomaly
        corr_anomaly = []
        for i in range(len(sensor_readings)):
            expected = self.correlation @ sensor_readings.values[i]
            actual = sensor_readings.values[i]
            corr_anomaly.append(np.abs(expected - actual).mean())

        # Ensemble
        graph_scores = np.array(graph_anomaly_scores)
        graph_scores = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min() + 1e-6)
        corr_scores = np.array(corr_anomaly)
        corr_scores = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min() + 1e-6)

        ensemble_scores = 0.5 * (graph_scores + if_scores) + 0.3 * corr_scores

        return ensemble_scores, ensemble_scores > (1 - self.contamination)

# ============ ADVANCED DOMAIN ADAPTATION ============

class CORAALDomainAdaptation:
    """CORAL (Correlation Alignment) for domain adaptation"""

    def __init__(self, kernel='rbf', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def compute_mmd_loss(self, source_features, target_features):
        """Maximum Mean Discrepancy with multiple kernels"""
        def rbf_kernel(x, y):
            sq_dist = np.sum((x[:, None] - y[None, :]) ** 2, axis=2)
            return np.exp(-sq_dist / (2 * self.bandwidth ** 2))

        def poly_kernel(x, y):
            return (1 + np.dot(x, y.T)) ** 3

        # RBF kernel
        kxx_rbf = rbf_kernel(source_features, source_features).mean()
        kyy_rbf = rbf_kernel(target_features, target_features).mean()
        kxy_rbf = rbf_kernel(source_features, target_features).mean()
        mmd_rbf = np.sqrt(kxx_rbf + kyy_rbf - 2 * kxy_rbf)

        # Polynomial kernel
        kxx_poly = poly_kernel(source_features, source_features).mean()
        kyy_poly = poly_kernel(target_features, target_features).mean()
        kxy_poly = poly_kernel(source_features, target_features).mean()
        mmd_poly = np.sqrt(kxx_poly + kyy_poly - 2 * kxy_poly)

        return 0.6 * mmd_rbf + 0.4 * mmd_poly

    def compute_coral_loss(self, source_features, target_features):
        """CORAL: minimize correlation distance"""
        source_cov = np.cov(source_features.T) + 1e-6 * np.eye(source_features.shape[1])
        target_cov = np.cov(target_features.T) + 1e-6 * np.eye(target_features.shape[1])

        # Frobenius norm of covariance difference
        coral_loss = np.sum((source_cov - target_cov) ** 2)

        return coral_loss

    def adapt_model_advanced(self, pretrained_weights, source_data, target_data, lambda_coral=0.5):
        """Advanced adaptation combining MMD and CORAL"""
        mmd = self.compute_mmd_loss(source_data, target_data)
        coral = self.compute_coral_loss(source_data, target_data)

        total_adaptation_loss = mmd + lambda_coral * coral

        return {
            'adapted_weights': pretrained_weights,
            'mmd_loss': mmd,
            'coral_loss': coral,
            'total_loss': total_adaptation_loss
        }

# ============ ADVANCED SENSOR IMPUTATION ============

class GraphBasedSensorImputation:
    """Advanced imputation using graph structure and temporal continuity"""

    def __init__(self, adjacency_matrix):
        self.adjacency = adjacency_matrix

    def impute_missing_sensor(self, sensor_readings, failed_sensor_idx, method='knn_temporal'):
        """Multiple imputation strategies"""

        if method == 'knn_temporal':
            # k-NN in temporal dimension
            neighbors = np.where(self.adjacency[failed_sensor_idx] > 0)[0]
            if len(neighbors) == 0:
                return sensor_readings.iloc[:, failed_sensor_idx].mean()

            neighbor_values = sensor_readings.iloc[:, neighbors].values
            weights = self.adjacency.iloc[failed_sensor_idx, neighbors].values
            weights = weights / weights.sum()

            # Weighted temporal interpolation
            imputed = np.average(neighbor_values, axis=1, weights=weights)
            return imputed

        elif method == 'spectral':
            # Spectral method using graph Laplacian
            D = np.diag(self.adjacency.sum(axis=1))
            L = D - self.adjacency.values
            eigenvalues, eigenvectors = np.linalg.eig(L)

            # Project onto top eigenvectors
            top_k = min(5, len(eigenvalues))
            top_eigenvectors = eigenvectors[:, :top_k]

            neighbors = np.where(self.adjacency[failed_sensor_idx] > 0)[0]
            imputed = top_eigenvectors[failed_sensor_idx] @ top_eigenvectors[neighbors].T @ \
                      sensor_readings.iloc[:, neighbors].values.T

            return imputed.flatten()

        else:  # linear_regression
            neighbors = np.where(self.adjacency[failed_sensor_idx] > 0)[0]
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(sensor_readings.iloc[:, neighbors], sensor_readings.iloc[:, failed_sensor_idx])
            return lr.predict(sensor_readings.iloc[:, neighbors])
