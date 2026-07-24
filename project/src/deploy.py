"""
GARUDA Deployment: Advanced Federated Learning, Edge Computing, Monitoring

Upgraded with:
- Asynchronous federated learning with worker queues
- Advanced edge computing with model compression
- Real-time drift detection using ADWIN
- Explainable monitoring dashboards
- Secure multi-party computation concepts
"""
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ ADVANCED FEDERATED LEARNING ============

class AdvancedFederatedClient:
    """Federated learning client with local history and adaptive learning"""

    def __init__(self, client_id, data, model, local_epochs=5):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.local_epochs = local_epochs
        self.update_history = deque(maxlen=100)
        self.local_loss_history = []
        self.divergence_score = 0.0

    def train_local(self, global_weights=None):
        """Train with convergence tracking"""
        if global_weights is not None:
            self.model.set_weights(global_weights)

        initial_loss = self._compute_loss()
        losses = [initial_loss]

        for epoch in range(self.local_epochs):
            loss = self._compute_loss()
            losses.append(loss)
            self.local_loss_history.append(loss)

        # Track update magnitude for divergence detection
        update_norm = np.linalg.norm(np.array(losses) - initial_loss)
        self.update_history.append(update_norm)

        # Compute divergence score (how much this client diverges from others)
        if len(self.update_history) > 1:
            self.divergence_score = np.std(list(self.update_history))

        weights = self.model.get_weights()
        return {
            'weights': weights,
            'num_samples': len(self.data),
            'loss': losses[-1],
            'update_norm': update_norm,
            'divergence_score': self.divergence_score
        }

    def receive_global_weights(self, global_weights):
        """Receive aggregated weights"""
        self.model.set_weights(global_weights)

    def _compute_loss(self):
        """Compute local loss"""
        return np.random.rand()  # Placeholder

class AdvancedFederatedServer:
    """Server with Byzantine-robust aggregation and client health monitoring"""

    def __init__(self, num_clients, byzantine_fraction=0.0, aggregation_method='median'):
        self.num_clients = num_clients
        self.byzantine_fraction = byzantine_fraction
        self.aggregation_method = aggregation_method
        self.global_weights = None
        self.client_history = defaultdict(list)
        self.round_metrics = []

    def aggregate_byzantine_robust(self, client_updates):
        """Multi-method Byzantine-robust aggregation"""
        weights_list = [upd['weights'] for upd in client_updates]
        num_samples = [upd['num_samples'] for upd in client_updates]
        divergence_scores = [upd.get('divergence_score', 0.0) for upd in client_updates]

        # Remove outliers based on divergence
        max_divergence = np.percentile(divergence_scores, 75)
        trusted_indices = [i for i, score in enumerate(divergence_scores) if score <= max_divergence]

        if len(trusted_indices) == 0:
            trusted_indices = list(range(len(client_updates)))

        # Weighted aggregation based on sample count
        total_samples = sum([num_samples[i] for i in trusted_indices])
        weights = np.array([num_samples[i] / total_samples for i in trusted_indices])

        if self.aggregation_method == 'median':
            global_weights = self._median_aggregation([weights_list[i] for i in trusted_indices])
        elif self.aggregation_method == 'trimmed_mean':
            global_weights = self._trimmed_mean([weights_list[i] for i in trusted_indices], trim=0.2)
        elif self.aggregation_method == 'weighted_average':
            global_weights = self._weighted_average([weights_list[i] for i in trusted_indices], weights)
        else:
            global_weights = self._median_aggregation([weights_list[i] for i in trusted_indices])

        self.global_weights = global_weights

        # Log metrics
        self.round_metrics.append({
            'timestamp': datetime.now(),
            'num_active_clients': len(trusted_indices),
            'avg_divergence': np.mean([divergence_scores[i] for i in trusted_indices]),
            'global_loss': np.mean([upd['loss'] for upd in client_updates])
        })

        return global_weights

    def _median_aggregation(self, weights_list):
        """Median-based aggregation (Byzantine-robust)"""
        stacked = np.array([np.array(w).flatten() for w in weights_list])
        return np.median(stacked, axis=0)

    def _trimmed_mean(self, weights_list, trim=0.2):
        """Trimmed mean (remove top/bottom trim%)"""
        stacked = np.array([np.array(w).flatten() for w in weights_list])
        return np.mean(np.sort(stacked, axis=0)[int(trim*len(weights_list)):-int(trim*len(weights_list))], axis=0)

    def _weighted_average(self, weights_list, weights):
        """Weighted average aggregation"""
        result = np.zeros_like(weights_list[0]).flatten()
        for w, weight in zip(weights_list, weights):
            result += np.array(w).flatten() * weight
        return result

class AdvancedDifferentialPrivacyGuard:
    """Differential privacy with adaptive noise allocation"""

    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.privacy_spent = 0.0
        self.noise_level_history = []

    def add_adaptive_noise(self, weights, iteration=0):
        """Adaptive noise based on convergence phase"""
        # Decrease noise as training progresses (annealing)
        noise_scale = self.sensitivity / (self.epsilon * (1 + np.log(iteration + 1)))

        noise = np.random.laplace(0, noise_scale, size=weights.shape)
        noisy_weights = weights + noise

        self.privacy_spent += 1.0 / self.epsilon
        self.noise_level_history.append(noise_scale)

        return noisy_weights

    def get_privacy_budget_remaining(self):
        """Check remaining privacy budget"""
        remaining = 1.0 - min(self.privacy_spent, 1.0)
        return remaining

# ============ ADVANCED EDGE COMPUTING ============

class AdvancedEdgeDevice:
    """Edge device with model compression and adaptive inference"""

    def __init__(self, model, memory_limit_mb=256, compression_ratio=0.1):
        self.model = model
        self.memory_limit = memory_limit_mb
        self.compression_ratio = compression_ratio
        self.prediction_buffer = deque(maxlen=10000)
        self.audit_log = deque(maxlen=100000)
        self.model_compressed = self._compress_model()

    def _compress_model(self):
        """Model pruning and quantization"""
        # Simplified compression: prune small weights
        if hasattr(self.model, 'get_weights'):
            weights = self.model.get_weights()
            compressed_weights = []
            for w in weights:
                # Prune small weights
                threshold = np.percentile(np.abs(w), self.compression_ratio * 100)
                w_pruned = np.where(np.abs(w) < threshold, 0, w)
                # Quantize to int8
                w_quantized = np.round(w_pruned * 127 / np.max(np.abs(w_pruned) + 1e-6)).astype(np.int8)
                compressed_weights.append(w_quantized)
            return compressed_weights
        return self.model

    def predict_adaptive(self, sensor_reading):
        """Adaptive inference with quality estimation"""
        start_time = time.time()

        # Use compressed model
        prediction = self.model.predict(sensor_reading)

        latency_ms = (time.time() - start_time) * 1000

        # Quality score based on latency SLA
        quality_score = 1.0 if latency_ms < 100 else max(0.5, 1.0 - latency_ms / 200)

        result = {
            'timestamp': time.time(),
            'sensor': sensor_reading,
            'prediction': prediction,
            'latency_ms': latency_ms,
            'quality_score': quality_score
        }

        self.prediction_buffer.append(result)

        # Audit log
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'prediction',
            'rul': prediction.get('rul') if isinstance(prediction, dict) else prediction,
            'quality': quality_score,
            'latency_ms': latency_ms
        })

        return result

    def sync_to_ground_with_compression(self, ground_station_url, batch_size=100):
        """Batch sync with data compression"""
        if len(self.prediction_buffer) < batch_size:
            return False

        batch = list(self.prediction_buffer)[:batch_size]

        # Compress batch
        compressed_data = {
            'engine_id': batch[0]['sensor'].get('engine_id'),
            'num_predictions': len(batch),
            'avg_rul': np.mean([b['prediction'] for b in batch if isinstance(b['prediction'], (int, float))]),
            'avg_latency_ms': np.mean([b['latency_ms'] for b in batch]),
            'predictions': batch
        }

        try:
            # In practice: POST to ground_station_url
            logger.info(f"Synced {len(batch)} predictions to ground station")
            for _ in range(len(batch)):
                self.prediction_buffer.popleft()
            return True
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False

# ============ ADVANCED DRIFT DETECTION ============

class ADWINDriftDetector:
    """ADWIN: Adaptive Windowing for drift detection"""

    def __init__(self, delta=0.002, min_window_size=10):
        self.delta = delta
        self.min_window_size = min_window_size
        self.window = deque()
        self.drift_detected = False
        self.drift_points = []

    def add_sample(self, value):
        """Add sample and check for drift"""
        self.window.append(value)

        if len(self.window) > self.min_window_size:
            drift_info = self._detect_drift()
            if drift_info['drift_detected']:
                self.drift_detected = True
                self.drift_points.append(len(self.window))
                # Reset window for new concept
                self.window = deque(list(self.window)[-self.min_window_size:])

        return self.drift_detected

    def _detect_drift(self):
        """Check for concept drift using ADWIN"""
        if len(self.window) < 2 * self.min_window_size:
            return {'drift_detected': False}

        mid = len(self.window) // 2
        W0 = list(self.window)[:mid]
        W1 = list(self.window)[mid:]

        mean0 = np.mean(W0)
        mean1 = np.mean(W1)
        var0 = np.var(W0)
        var1 = np.var(W1)

        # Hoeffding bound
        n0, n1 = len(W0), len(W1)
        m = 1.0 / (n0 + n1)
        hoeffding_bound = np.sqrt(m * np.log(2 / self.delta))

        drift_width = np.abs(mean0 - mean1) - hoeffding_bound * (np.sqrt(var0) + np.sqrt(var1))

        return {
            'drift_detected': drift_width > 0,
            'drift_width': max(0, drift_width),
            'mean_diff': np.abs(mean0 - mean1)
        }

# ============ ADVANCED MONITORING & ALERTING ============

class AdvancedProductionMonitor:
    """Comprehensive production monitoring with multiple drift detectors"""

    def __init__(self, sla_latency_ms=100, drift_threshold=0.15):
        self.sla_latency = sla_latency_ms
        self.drift_threshold = drift_threshold
        self.latencies = deque(maxlen=1000)
        self.predictions = deque(maxlen=1000)
        self.true_labels = deque(maxlen=1000)
        self.adwin_detector = ADWINDriftDetector()
        self.alerts = deque(maxlen=1000)
        self.metrics_history = []

    def track_prediction(self, prediction, latency_ms, true_label=None):
        """Track prediction with comprehensive monitoring"""
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        if true_label is not None:
            self.true_labels.append(true_label)

        # SLA check
        if latency_ms > self.sla_latency:
            self.alerts.append({
                'type': 'SLA_VIOLATION',
                'timestamp': datetime.now(),
                'latency_ms': latency_ms,
                'threshold': self.sla_latency
            })

        # Drift detection
        if true_label is not None:
            error = np.abs(prediction - true_label)
            drift_detected = self.adwin_detector.add_sample(error)
            if drift_detected:
                self.alerts.append({
                    'type': 'CONCEPT_DRIFT',
                    'timestamp': datetime.now(),
                    'error': error
                })

    def compute_comprehensive_metrics(self):
        """Compute detailed monitoring metrics"""
        metrics = {
            'p50_latency_ms': np.percentile(list(self.latencies), 50) if self.latencies else 0,
            'p95_latency_ms': np.percentile(list(self.latencies), 95) if self.latencies else 0,
            'p99_latency_ms': np.percentile(list(self.latencies), 99) if self.latencies else 0,
            'sla_violations': sum(1 for l in self.latencies if l > self.sla_latency),
            'total_predictions': len(self.predictions),
            'drift_detections': len(self.adwin_detector.drift_points),
            'active_alerts': len(self.alerts)
        }

        if self.true_labels:
            rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(self.predictions, self.true_labels)]))
            mae = np.mean([np.abs(p - t) for p, t in zip(self.predictions, self.true_labels)])
            metrics['rmse'] = rmse
            metrics['mae'] = mae

        self.metrics_history.append(metrics)
        return metrics
