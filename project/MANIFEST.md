# GARUDA - Module Manifest

## Core (13 modules)
Essential ML pipeline for turbojet health monitoring.

| Module | Purpose |
|--------|---------|
| ingestion | Data preprocessing & 60/20/20 split |
| aggregation | Health index estimation & EHI weighting |
| quantification | Conformal prediction & RUL bounds |
| validation | Safety certification & bounds checking |
| diagnostics | Sensor validation & integrity |
| adaptation | Adaptive sensor envelope (Bayesian) |
| regression | Heteroscedastic variance modeling |
| inference | Physics-informed neural networks |
| transfer | Domain adaptation (MMD) |
| multitask | Multi-task health learning |
| topology | GraphSage sensor diagnosis |
| dynamics | Neural ODE degradation solver |
| attribution | Causal health explanation |
| security | HMAC, JWT, RBAC, AES encryption |

## Test (6 modules)
Comprehensive validation suite.

| Module | Purpose |
|--------|---------|
| robustness | Adversarial & extreme condition tests |
| anomaly | Anomaly injection & detection |
| calibration | Model calibration analysis |
| generalization | Out-of-distribution testing |
| propagation | Uncertainty propagation |
| validation | Master test orchestrator |

## Deploy (4 modules)
Production deployment systems.

| Module | Purpose |
|--------|---------|
| federation | Multi-airline federated learning |
| telemetry | On-board edge deployment |
| monitoring | Dashboard & monitoring |
| interpretation | Feature importance & explainability |

## India (3 modules)
India-specific market features.

| Module | Purpose |
|--------|---------|
| compliance | DGCA maintenance release |
| optimization | Cost & ROI calculator |
| profiles | Airline-specific configurations |

## Utils (1 module)
Utilities.

| Module | Purpose |
|--------|---------|
| benchmark | C-MAPSS dataset loader |

---

**Total:** 28 professional modules | 7,500+ lines | Production-ready
