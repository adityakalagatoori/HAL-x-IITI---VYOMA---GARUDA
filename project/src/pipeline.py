"""
GARUDA Pipeline: Advanced Data Ingestion, Aggregation, Quantification, Validation

Upgraded with:
- Advanced feature engineering (wavelet, FFT, statistical)
- Robust outlier detection
- Sensor quality scoring
- Advanced topology analysis
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import signal, stats
from pydantic import BaseModel, Field, validator
import warnings

warnings.filterwarnings('ignore')

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
    sensor_quality: float = Field(ge=0, le=1, default=1.0)

class HealthAssessment(BaseModel):
    compressor_health: float = Field(ge=0, le=100)
    combustor_health: float = Field(ge=0, le=100)
    turbine_health: float = Field(ge=0, le=100)
    overall_health: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)

    @validator('overall_health')
    def health_monotonic(cls, v, values):
        if 'compressor_health' in values:
            avg = (values['compressor_health'] + values.get('combustor_health', 50) + values.get('turbine_health', 50)) / 3
            assert abs(v - avg) < 50, "Overall health inconsistent"
        return v

class AuditEvent(BaseModel):
    timestamp: str
    user: str
    action: str
    engine_id: str
    details: dict

# ============ ADVANCED FEATURE ENGINEERING ============

class AdvancedFeatureEngineer:
    """Multi-scale temporal and spectral feature extraction"""

    def __init__(self, windows=[5, 10, 20, 50]):
        self.windows = windows
        self.scaler = RobustScaler()

    def rolling_statistics(self, data):
        """Compute rolling statistics across multiple windows"""
        features = {}
        for window in self.windows:
            for col in data.columns:
                if data[col].dtype in [np.float64, np.int64]:
                    features[f'{col}_mean_{window}'] = data[col].rolling(window).mean()
                    features[f'{col}_std_{window}'] = data[col].rolling(window).std()
                    features[f'{col}_min_{window}'] = data[col].rolling(window).min()
                    features[f'{col}_max_{window}'] = data[col].rolling(window).max()
                    features[f'{col}_skew_{window}'] = data[col].rolling(window).skew()
                    features[f'{col}_kurtosis_{window}'] = data[col].rolling(window).kurt()
        return pd.DataFrame(features).fillna(method='bfill')

    def spectral_features(self, data, sampling_rate=1):
        """Extract FFT-based spectral features"""
        features = {}
        for col in data.select_dtypes(np.number).columns:
            signal_data = data[col].dropna().values
            if len(signal_data) > 10:
                fft = np.fft.fft(signal_data)
                power = np.abs(fft) ** 2
                features[f'{col}_power_mean'] = np.mean(power)
                features[f'{col}_power_std'] = np.std(power)
                features[f'{col}_power_peak'] = np.max(power)

                # Dominant frequency
                freqs = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
                dominant_freq_idx = np.argmax(power)
                features[f'{col}_dominant_freq'] = freqs[dominant_freq_idx]

        return pd.DataFrame({k: [v] for k, v in features.items()})

    def wavelet_features(self, data):
        """Decompose signal into wavelet components"""
        features = {}
        for col in data.select_dtypes(np.number).columns:
            signal_data = data[col].dropna().values
            if len(signal_data) > 32:
                # Discrete wavelet transform (Daubechies)
                coeffs = signal.wavedec(signal_data, 'db4', level=3)
                for i, coeff in enumerate(coeffs):
                    features[f'{col}_wavelet_l{i}_mean'] = np.mean(coeff)
                    features[f'{col}_wavelet_l{i}_energy'] = np.sum(coeff**2)

        return pd.DataFrame({k: [v] for k, v in features.items()})

    def rate_of_change_features(self, data):
        """Compute derivatives and second derivatives"""
        features = {}
        for col in data.select_dtypes(np.number).columns:
            features[f'{col}_delta'] = data[col].diff()  # First derivative
            features[f'{col}_delta2'] = data[col].diff().diff()  # Second derivative

        return pd.DataFrame(features).fillna(method='bfill')

# ============ ROBUST OUTLIER DETECTION ============

class OutlierDetector:
    """Multi-method outlier detection with confidence scoring"""

    def __init__(self, contamination=0.05):
        self.contamination = contamination

    def isolation_forest_score(self, data):
        """Isolation Forest anomaly scoring"""
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        scores = iso_forest.fit_predict(data)
        return (scores + 1) / 2  # Convert [-1, 1] to [0, 1]

    def mahalanobis_distance(self, data):
        """Mahalanobis distance-based outlier detection"""
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

        distances = []
        for sample in data:
            diff = sample - mean
            distance = np.sqrt(diff @ inv_cov @ diff.T)
            distances.append(distance)

        # Normalize to [0, 1]
        distances = np.array(distances)
        return 1 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)

    def local_outlier_factor(self, data):
        """Local Outlier Factor-based detection"""
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        scores = lof.fit_predict(data)
        lof_scores = -lof.negative_outlier_factor_
        return 1 - (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-6)

    def ensemble_outlier_score(self, data):
        """Ensemble of multiple outlier detection methods"""
        iso_scores = self.isolation_forest_score(data)
        maha_scores = self.mahalanobis_distance(data)
        lof_scores = self.local_outlier_factor(data)

        # Weighted ensemble
        ensemble = 0.4 * iso_scores + 0.3 * maha_scores + 0.3 * lof_scores
        return ensemble

# ============ ADVANCED PIPELINE ============

def load_split_with_proper_division(test_size=0.2, val_size=0.25):
    """Load C-MAPSS with stratified splitting and advanced validation"""
    train_official = pd.read_csv("../data/train.csv")
    test_official = pd.read_csv("../data/test.csv")
    gt = pd.read_csv("../data/ground_truth.csv")

    train_official = train_official.merge(gt, on=["EngineID", "Cycle"], how='inner')
    test_official = test_official.merge(gt, on=["EngineID", "Cycle"], how='inner')

    # Stratified split on engine ID
    engines = train_official['EngineID'].unique()
    train_engines, val_engines = train_test_split(engines, test_size=val_size, random_state=42)

    train_sub = train_official[train_official['EngineID'].isin(train_engines)]
    val_sub = train_official[train_official['EngineID'].isin(val_engines)]

    return train_sub, val_sub, test_official

def aggregate_by_engine_advanced(data):
    """Advanced multi-scale aggregation with feature engineering"""
    engineer = AdvancedFeatureEngineer(windows=[5, 10, 20, 50])

    # Rolling statistics
    rolling_feats = engineer.rolling_statistics(data)

    # Rate of change features
    rate_feats = engineer.rate_of_change_features(data)

    # Spectral features
    spectral_feats = engineer.spectral_features(data)

    # Wavelet features
    wavelet_feats = engineer.wavelet_features(data)

    # Combine all features
    all_features = pd.concat([rolling_feats, rate_feats, spectral_feats, wavelet_feats], axis=1)

    return all_features.fillna(method='bfill').dropna()

def calculate_health_metrics_advanced(sensor_data):
    """Advanced health metric calculation with uncertainty"""
    data = sensor_data.copy()

    # Pressure ratios with smoothing
    pressure_ratio = data['P3'] / (data['P2'] + 1e-6)
    pressure_ratio_smooth = pd.Series(pressure_ratio).rolling(5, center=True).mean()

    # Temperature rise with normalization
    temp_rise = data['T3'] - data['T2']
    temp_rise_norm = (temp_rise - temp_rise.min()) / (temp_rise.max() - temp_rise.min() + 1e-6)

    # Turbine power extraction
    turbine_power = data['P3'] * (data['T4'] - data['T3'])
    turbine_power_norm = (turbine_power - turbine_power.min()) / (turbine_power.max() - turbine_power.min() + 1e-6)

    # Health metrics with confidence scores
    data['CompressorHealth'] = 100 * np.clip(pressure_ratio_smooth / pressure_ratio_smooth.max(), 0, 1)
    data['CombustorHealth'] = 100 * np.clip(temp_rise_norm, 0, 1)
    data['TurbineHealth'] = 100 * np.clip(turbine_power_norm, 0, 1)

    # Weighted overall health
    data['OverallHealth'] = 0.35 * data['CompressorHealth'] + \
                            0.35 * data['CombustorHealth'] + \
                            0.30 * data['TurbineHealth']

    # Confidence based on stability
    compressor_confidence = 1 - np.abs(data['CompressorHealth'].diff()).mean() / 100
    data['Confidence'] = np.clip(compressor_confidence, 0.7, 1.0)

    return data[['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Confidence']]

def validate_all_advanced(data):
    """Advanced validation with sensor quality scoring"""
    validated = []
    outlier_detector = OutlierDetector()

    numeric_cols = data.select_dtypes(np.number).columns
    sensor_quality_scores = outlier_detector.ensemble_outlier_score(data[numeric_cols].fillna(data.mean()))

    for idx, row in data.iterrows():
        try:
            reading = SensorReading(
                **row.to_dict(),
                sensor_quality=sensor_quality_scores[idx]
            )
            validated.append(reading)
        except Exception as e:
            print(f"Validation error on row {idx}: {e}")

    return validated

def build_sensor_topology_advanced(data, correlation_threshold=0.6):
    """Advanced topology analysis with hierarchical clustering"""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    numeric_cols = data.select_dtypes(np.number).columns
    correlation_matrix = data[numeric_cols].corr().abs()

    # Build adjacency from correlation
    adjacency = (correlation_matrix > correlation_threshold).astype(int)
    np.fill_diagonal(adjacency.values, 0)

    # Hierarchical clustering of sensors
    if len(numeric_cols) > 1:
        distance_matrix = 1 - correlation_matrix.values
        try:
            linkage_matrix = linkage(pdist(distance_matrix, metric='euclidean'), method='ward')
        except:
            linkage_matrix = None
    else:
        linkage_matrix = None

    return {
        'adjacency': adjacency,
        'correlation': correlation_matrix,
        'linkage': linkage_matrix
    }

def compute_sensor_importance_advanced(topology_dict):
    """Advanced centrality analysis"""
    adjacency = topology_dict['adjacency']

    # Degree centrality
    degree_centrality = adjacency.sum(axis=1) / (len(adjacency) - 1)

    # Betweenness centrality approximation
    betweenness = np.zeros(len(adjacency))
    for i in range(len(adjacency)):
        neighbors = np.where(adjacency.iloc[i] > 0)[0]
        betweenness[i] = len(neighbors) * np.mean([adjacency.iloc[i, j] for j in neighbors]) if len(neighbors) > 0 else 0

    # Eigenvector centrality approximation (power iteration)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(adjacency.values)
        eigenvector_cent = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
        eigenvector_cent = eigenvector_cent / eigenvector_cent.sum()
    except:
        eigenvector_cent = np.ones(len(adjacency)) / len(adjacency)

    # Combined importance score
    importance = 0.4 * degree_centrality + 0.3 * betweenness + 0.3 * eigenvector_cent

    return pd.Series(importance, index=adjacency.index).sort_values(ascending=False)
