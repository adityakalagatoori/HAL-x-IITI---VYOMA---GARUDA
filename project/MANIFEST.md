# GARUDA Manifest — 7 Condensed Modules

## Module Dependency Graph

```
data/ (8 CSV files)
  ↓
pipeline.py (load, aggregate, quantify, validate)
  ↓↙─────────────────┐
physics.py           predict.py
(PINN, ODE,         (RUL, multitask,
 anomaly, adapt)     transfer, SHAP)
  ↓                  ↓
  └────────┬─────────┘
           ↓
        deploy.py
        (federated, edge, monitor, explain)
           ↓
        india.py
        (compliance, ROI, profiles)
           ↓
        test.py (robustness, calibration, generalization)
           ↓
        benchmark.py (latency, throughput, memory)
```

## Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **pipeline.py** | Data ingestion, aggregation, quantification, validation | `load_split()`, `aggregate_by_engine()`, `calculate_health_metrics()`, `validate_all()`, `build_sensor_topology()` |
| **physics.py** | PINN, neural ODE, anomaly detection, domain adaptation | `DeepONetPINN`, `DegradationDynamics`, `GraphSageAnomalyDetector`, `MaximumMeanDiscrepancy` |
| **predict.py** | RUL prediction, multitask, transfer learning, SHAP | `ConformalRULPredictor`, `MultiTaskNetwork`, `TransferLearning`, `CausalExplainability` |
| **deploy.py** | Federated learning, edge deployment, monitoring, interpretation | `FederatedClient/Server`, `EdgeDevice`, `ProductionMonitor`, `ExplainabilityEngine` |
| **india.py** | DGCA compliance, cost-ROI, airline profiles, security | `DGCAComplianceReporter`, `MaintenanceCostCalculator`, `AirlineCustomizer`, `SecureDigitalSignature` |
| **test.py** | Robustness, anomaly, calibration, generalization, propagation, statistical validation | 12 test functions covering all aspects |
| **benchmark.py** | Latency, throughput, memory, SLA compliance | `PerformanceBenchmark` class with 4 methods |

## Data Flow

`project/data/*.csv` → pipeline.py → physics.py → predict.py → deploy.py → india.py → test.py → benchmark.py

## Integration

All modules are loosely coupled with clear input/output contracts.  
Cross-cutting concerns: Security (india.py), Monitoring (deploy.py), Testing (test.py)

---

**7 condensed modules | ~4,500 LOC | Production-ready for Aerothon 2026**
