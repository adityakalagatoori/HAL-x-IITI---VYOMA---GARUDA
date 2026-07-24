"""
GARUDA Pipeline: Data Ingestion, Aggregation, Quantification, Validation

Combines: ingestion.py, aggregation.py, quantification.py, validation.py, topology.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pydantic import BaseModel, Field, validator

# ============ PYDANTIC VALIDATORS ============
class SensorReading(BaseModel):
    engine_id: str
    cycle: int
    altitude_m: float = Field(ge=0, le=45000)
    mach: float = Field(ge=0, le=2.5)
    tamb_k: float = Field(ge=200, le=350)
    pamb_pa: float = Field(ge=10000, le=101325)
    rpm_rev_min: float = Field(ge=0, le=50000)
    fuel_flow_kg_s: float = Field(ge=0, le=10)
    p2_pa: float = Field(ge=10000, le=1000000)
    t2_k: float = Field(ge=200, le=1000)
    p3_pa: float = Field(ge=100000, le=10000000)
    t3_k: float = Field(ge=500, le=2000)
    p4_pa: float = Field(ge=100000, le=5000000)
    t4_k: float = Field(ge=800, le=2500)

class HealthAssessment(BaseModel):
    compressor_health: float = Field(ge=0, le=100)
    combustor_health: float = Field(ge=0, le=100)
    turbine_health: float = Field(ge=0, le=100)
    overall_health: float = Field(ge=0, le=100)

    @validator('overall_health')
    def health_monotonic(cls, v, values):
        if 'compressor_health' in values:
            avg = (values['compressor_health'] + values.get('combustor_health', 50) + values.get('turbine_health', 50)) / 3
            assert abs(v - avg) < 50, "Overall health inconsistent with component healths"
        return v

class AuditEvent(BaseModel):
    timestamp: str
    user: str
    action: str
    engine_id: str
    details: dict

# ============ PIPELINE FUNCTIONS ============

def load_split_with_proper_division():
    """Load NASA C-MAPSS + synthetic labels with proper 60/20/20 split."""
    train_official = pd.read_csv("../data/train.csv")
    test_official = pd.read_csv("../data/test.csv")
    gt = pd.read_csv("../data/ground_truth.csv")

    train_official = train_official.merge(gt, on=["EngineID", "Cycle"])
    test_official = test_official.merge(gt, on=["EngineID", "Cycle"])

    train_sub, val_sub = train_test_split(train_official, test_size=0.25, random_state=42)

    return train_sub, val_sub, test_official

def aggregate_by_engine(data, windows=[5, 10, 20]):
    """Temporal aggregation: rolling statistics at multiple time scales."""
    aggregated = []
    for window in windows:
        rolling_mean = data.rolling(window=window, center=True).mean()
        rolling_std = data.rolling(window=window, center=True).std()
        aggregated.extend([rolling_mean, rolling_std])
    return pd.concat(aggregated, axis=1)

def calculate_health_metrics(sensor_data):
    """Derive health scores (0-100) from raw sensors."""
    data = sensor_data.copy()

    # Compressor health: based on pressure ratio (P3/P2)
    p3_p2_ratio = data['P3'] / data['P2']
    data['CompressorHealth'] = 100 * np.clip(p3_p2_ratio / p3_p2_ratio.max(), 0, 1)

    # Combustor health: based on temperature rise (T3-T2)
    temp_rise = data['T3'] - data['T2']
    data['CombustorHealth'] = 100 * np.clip(temp_rise / temp_rise.max(), 0, 1)

    # Turbine health: based on power extraction
    turbine_power = data['P3'] * (data['T4'] - data['T3'])
    data['TurbineHealth'] = 100 * np.clip(turbine_power / turbine_power.max(), 0, 1)

    # Overall: weighted average
    data['OverallHealth'] = 0.4 * data['CompressorHealth'] + \
                            0.3 * data['CombustorHealth'] + \
                            0.3 * data['TurbineHealth']

    return data[['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth']]

def validate_all(data):
    """Pydantic validation: ensure type safety and business rules."""
    validated = []
    for _, row in data.iterrows():
        try:
            reading = SensorReading(**row.to_dict())
            validated.append(reading)
        except Exception as e:
            print(f"Validation error on row: {e}")
    return validated

def build_sensor_topology(correlation_matrix, threshold=0.7):
    """Build sensor network graph from correlation matrix."""
    adjacency = (correlation_matrix.abs() > threshold).astype(int)
    np.fill_diagonal(adjacency.values, 0)
    return adjacency

def compute_sensor_importance(adjacency_matrix):
    """Compute sensor centrality (importance in network)."""
    degree_centrality = adjacency_matrix.sum(axis=1) / (len(adjacency_matrix) - 1)
    return degree_centrality.sort_values(ascending=False)
