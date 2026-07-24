"""
GARUDA Deployment: Federated Learning, Edge, Monitoring, Interpretation

Combines: federation.py, telemetry.py, monitoring.py, interpretation.py
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import logging

# ============ FEDERATED LEARNING ============

class FederatedClient:
    """Local training client (one airline)."""
    def __init__(self, client_id, data, model):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.weights = None

    def train_local(self, epochs=5):
        """Train locally on own data."""
        # Simplified: in practice use SGD on batches
        self.model.fit(self.data['X'], self.data['y'], epochs=epochs)
        self.weights = self.model.get_weights()
        return self.weights

    def receive_global_weights(self, global_weights):
        """Receive aggregated weights from server."""
        self.model.set_weights(global_weights)

class FederatedServer:
    """Central aggregator (ground station)."""
    def __init__(self, num_clients, byzantine_fraction=0.0):
        self.num_clients = num_clients
        self.byzantine_fraction = byzantine_fraction
        self.global_weights = None

    def aggregate_byzantine_robust(self, client_weights):
        """Median aggregation: robust to 50% poisoned clients."""
        # Stack all weights
        weight_stack = np.array(client_weights)
        # Compute median per weight (robust to outliers)
        global_weights = np.median(weight_stack, axis=0)
        self.global_weights = global_weights
        return global_weights

class DifferentialPrivacyGuard:
    """Add noise for privacy (ε-δ privacy budget)."""
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = epsilon

    def add_noise_to_weights(self, weights):
        """Add Gaussian noise to weights."""
        noise_scale = 1.0 / self.epsilon
        noise = np.random.normal(0, noise_scale, size=weights.shape)
        noisy_weights = weights + noise
        self.privacy_budget -= (1.0 / self.epsilon)  # Account for privacy cost
        return noisy_weights

# ============ EDGE DEPLOYMENT ============

class EdgeDevice:
    """Onboard aircraft computer (ARM, 512MB RAM)."""
    def __init__(self, model, memory_limit_mb=256):
        self.model = model
        self.memory_limit = memory_limit_mb
        self.prediction_buffer = []
        self.audit_log = []
        self.max_buffer_size = 1000

    def predict(self, sensor_reading):
        """Real-time prediction (no latency)."""
        prediction = self.model.predict(sensor_reading)
        self.prediction_buffer.append({
            'timestamp': time.time(),
            'sensor': sensor_reading,
            'prediction': prediction
        })
        # Log to audit trail
        self.audit_log.append({
            'timestamp': time.time(),
            'action': 'prediction',
            'rul': prediction.get('rul')
        })
        return prediction

    def sync_to_ground(self, ground_station_url):
        """Intermittent async sync to ground station."""
        if len(self.prediction_buffer) == 0:
            return
        # In practice: POST to ground station API
        # Handle intermittent connectivity
        try:
            # Upload predictions, audit log
            self.prediction_buffer = []
            self.audit_log = []
        except Exception as e:
            # Retry on next sync
            logging.error(f"Sync failed: {e}")

# ============ PRODUCTION MONITORING ============

class ProductionMonitor:
    """Track SLA, detect drift, trigger retraining."""
    def __init__(self, sla_latency_ms=100, drift_threshold=0.15):
        self.sla_latency = sla_latency_ms
        self.drift_threshold = drift_threshold
        self.latencies = []
        self.predictions = []
        self.true_labels = []
        self.alerts = []

    def track_prediction(self, prediction, latency_ms, true_label=None):
        """Track each prediction."""
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        if true_label is not None:
            self.true_labels.append(true_label)

        # Check SLA
        if latency_ms > self.sla_latency:
            self.alerts.append(f"SLA violation: latency {latency_ms}ms > {self.sla_latency}ms")

    def detect_covariate_shift(self):
        """Detect input distribution shift."""
        # Simplified: compare recent vs historical distribution
        if len(self.predictions) < 100:
            return False
        recent_mean = np.mean(self.predictions[-50:])
        historical_mean = np.mean(self.predictions[:-50])
        shift = abs(recent_mean - historical_mean) / (historical_mean + 1e-6)
        return shift > self.drift_threshold

    def detect_label_shift(self):
        """Detect output distribution shift."""
        if len(self.true_labels) < 100:
            return False
        recent_mean = np.mean(self.true_labels[-50:])
        historical_mean = np.mean(self.true_labels[:-50])
        shift = abs(recent_mean - historical_mean) / (historical_mean + 1e-6)
        return shift > self.drift_threshold

    def compute_sla_metrics(self):
        """P50, P95, P99 latencies."""
        return {
            'p50': np.percentile(self.latencies, 50),
            'p95': np.percentile(self.latencies, 95),
            'p99': np.percentile(self.latencies, 99),
            'alerts': len(self.alerts)
        }

# ============ EXPLAINABILITY & INTERPRETATION ============

class ExplainabilityEngine:
    """Generate pilot-facing summaries."""
    def __init__(self, shap_values=None, feature_names=None):
        self.shap_values = shap_values
        self.feature_names = feature_names

    def generate_pilot_summary(self, rul_prediction, health_breakdown):
        """Human-readable summary for pilot."""
        summary = f"""
GARUDA Health Report
====================
Remaining Useful Life (RUL): {rul_prediction['rul']:.1f} hours
Confidence: ±{rul_prediction['upper'] - rul_prediction['rul']:.1f} hours (90%)

Component Health:
- Compressor: {health_breakdown['compressor']:.0f}%
- Combustor: {health_breakdown['combustor']:.0f}%
- Turbine: {health_breakdown['turbine']:.0f}%
- Overall: {health_breakdown['overall']:.0f}%

Recommendation:
"""
        if rul_prediction['rul'] < 50:
            summary += "🔴 CRITICAL: Schedule maintenance within 48 hours\n"
        elif rul_prediction['rul'] < 150:
            summary += "🟡 WARNING: Plan maintenance within 1 week\n"
        else:
            summary += "🟢 HEALTHY: Continue normal operations\n"

        return summary

    def generate_causal_chain(self, top_contributing_factors):
        """Explain WHY this engine is degrading."""
        chain = "Degradation Chain:\n"
        for i, factor in enumerate(top_contributing_factors, 1):
            chain += f"{i}. {factor['component']} efficiency ↓ by {factor['amount']:.1f}%\n"
            chain += f"   → Caused by {factor['root_cause']}\n"
            chain += f"   → Impact on RUL: -{factor['rul_impact']:.1f} hours\n"
        return chain

    def attention_weights_summary(self, attention_weights, timesteps):
        """Which timesteps matter most for this prediction."""
        top_timesteps = np.argsort(attention_weights)[-5:][::-1]
        summary = "Key Timesteps (most important for this prediction):\n"
        for t in top_timesteps:
            summary += f"  - Hour {timesteps[t]}: {attention_weights[t]:.2%} importance\n"
        return summary
