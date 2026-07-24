"""
GARUDA Benchmark: Performance Profiling, Throughput, Latency
"""
import time
import numpy as np
import psutil
import os

class PerformanceBenchmark:
    """Measure inference latency, throughput, memory."""
    def __init__(self, model):
        self.model = model
        self.latencies = []
        self.memory_usage = []

    def benchmark_latency(self, X_test, num_runs=1000):
        """Measure inference time."""
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.model.predict(X_test[[0]])  # Single sample
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        return {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'mean': np.mean(latencies),
            'std': np.std(latencies)
        }

    def benchmark_throughput(self, X_test, batch_size=32, duration_seconds=10):
        """Measure predictions per second."""
        process = psutil.Process(os.getpid())
        predictions_count = 0

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size]
                self.model.predict(batch)
                predictions_count += len(batch)

        elapsed = time.time() - start_time
        throughput = predictions_count / elapsed

        return {
            'predictions_per_second': throughput,
            'predictions_total': predictions_count,
            'elapsed_seconds': elapsed
        }

    def benchmark_memory(self, X_test):
        """Measure memory usage."""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run predictions
        for i in range(0, min(1000, len(X_test)), 100):
            self.model.predict(X_test[i:i+100])

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before

        return {
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'delta_mb': mem_delta
        }

    def sla_compliance_check(self, latencies_ms, sla_threshold_ms=100):
        """Check if P95 latency meets SLA."""
        p95 = np.percentile(latencies_ms, 95)
        compliant = p95 <= sla_threshold_ms
        return {
            'p95_latency_ms': p95,
            'sla_threshold_ms': sla_threshold_ms,
            'compliant': compliant
        }
