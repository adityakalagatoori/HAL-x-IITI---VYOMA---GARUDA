# Data Provenance

## Official (NASA C-MAPSS)
- **train.csv** — 54 KB — Training data
- **test.csv** — 14 KB — Held-out evaluation

## Generated (GARUDA Model)
- **ground_truth.csv** — 36 KB — RUL labels (synthetic)
- **physics_layer_output.csv** — 122 KB — PINN features
- **stage_b_predictions.csv** — 23 KB — v1 predictions
- **stage_b_v2_predictions.csv** — 19 KB — v2 predictions
- **stage_b_v3_predictions.csv** — 9.4 KB — v3 predictions

## Hybrid (Official + GARUDA)
- **turbojet_complete_dataset.csv** — 101 KB — C-MAPSS sensors + health metrics

**Total:** 8 files, 379 KB, 2 official + 5 generated + 1 hybrid
