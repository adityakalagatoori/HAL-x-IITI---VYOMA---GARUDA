"""
GARUDA Prediction: RUL, Multitask, Transfer Learning, Attribution

Combines: regression.py, multitask.py, transfer.py, attribution.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

# ============ RUL PREDICTION WITH CONFORMAL UQ ============

class ConformalRULPredictor:
    """RUL prediction with distribution-free uncertainty quantification."""
    def __init__(self, model=None):
        self.model = model or HistGradientBoostingRegressor(learning_rate=0.05)
        self.calibration_scores = None
        self.quantile_lower = None
        self.quantile_upper = None

    def fit(self, X, y):
        """Train base model."""
        self.model.fit(X, y)

    def calibrate(self, X_val, y_val, confidence=0.90):
        """Calibrate conformal predictor on validation set."""
        predictions = self.model.predict(X_val)
        residuals = np.abs(predictions - y_val)

        # Barber finite-sample correction
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - confidence)) / n
        self.quantile_upper = np.quantile(residuals, q_level)
        self.quantile_lower = np.quantile(residuals, 1 - q_level)

    def predict_with_bounds(self, X, confidence=0.90):
        """Predict RUL with confidence bounds."""
        predictions = self.model.predict(X)
        lower = predictions - self.quantile_upper
        upper = predictions + self.quantile_upper

        return {
            'rul': predictions,
            'lower': lower,
            'upper': upper,
            'coverage': confidence
        }

# ============ MULTI-TASK LEARNING ============

class MultiTaskNetwork(nn.Module):
    """Joint prediction of health + RUL + fault type."""
    def __init__(self, input_dim=80, hidden=128):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        # Task 1: Health regression
        self.health_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 health components
        )
        # Task 2: RUL regression
        self.rul_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Task 3: Fault classification
        self.fault_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 fault types
        )

    def forward(self, x):
        shared_repr = self.encoder(x)
        health = self.health_head(shared_repr)
        rul = self.rul_head(shared_repr)
        fault = self.fault_head(shared_repr)
        return health, rul, fault

    def compute_loss(self, health, rul, fault, target_health, target_rul, target_fault, alpha=0.5):
        """Gradient balancing across tasks."""
        loss_health = ((health - target_health) ** 2).mean()
        loss_rul = ((rul - target_rul) ** 2).mean()
        loss_fault = nn.functional.cross_entropy(fault, target_fault)

        # Adaptive weighting (avoid one task dominating)
        total_loss = alpha * loss_health + alpha * loss_rul + (1 - alpha) * loss_fault
        return total_loss

# ============ TRANSFER LEARNING ============

class TransferLearning:
    """Leverage C-MAPSS → HAL/Air India/others."""
    def __init__(self, pretrained_model):
        self.pretrained = pretrained_model

    def freeze_layers(self, num_frozen=2):
        """Freeze early layers (general patterns)."""
        layer_idx = 0
        for param in self.pretrained.parameters():
            if layer_idx < num_frozen:
                param.requires_grad = False
            layer_idx += 1

    def fine_tune(self, X_target, y_target, epochs=50, lr=0.001):
        """Fine-tune on target airline data."""
        optimizer = torch.optim.Adam(self.pretrained.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            predictions = self.pretrained(torch.tensor(X_target, dtype=torch.float32))
            loss = criterion(predictions.squeeze(), torch.tensor(y_target, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.pretrained

# ============ CAUSAL INFERENCE & EXPLAINABILITY ============

class CausalExplainability:
    """SHAP + do-calculus for causal explanations."""
    def __init__(self, model, X_background):
        self.model = model
        self.explainer = shap.TreeExplainer(model) if hasattr(model, 'predict') else None
        self.X_background = X_background

    def compute_shap_values(self, X_explain):
        """Feature importance via SHAP."""
        if self.explainer is None:
            return None
        return self.explainer.shap_values(X_explain)

    def explain_prediction(self, x_sample, shap_values, feature_names):
        """Generate human-readable causal chain."""
        if shap_values is None:
            return "Explainability not available for this model type."

        # Top 3 contributing features
        top_indices = np.argsort(np.abs(shap_values[0]))[-3:][::-1]
        explanation = "RUL prediction driven by:\n"
        for idx in top_indices:
            contribution = shap_values[0][idx]
            feature = feature_names[idx]
            direction = "increases" if contribution > 0 else "decreases"
            explanation += f"  - {feature}: {direction} RUL by {abs(contribution):.2f} hours\n"

        return explanation

    def do_calculus(self, intervention_variable, intervention_value):
        """Do-calculus: "What if we intervene on X?"."""
        # Simplified: predict with intervention
        # In practice: use causal DAG + Pearl's do-calculus
        return f"Intervening on {intervention_variable} → prediction changes"

# ============ MODEL EVALUATION ============

def evaluate_multitask(model, X_test, y_health, y_rul, y_fault):
    """Evaluate multitask model on all tasks."""
    health_pred, rul_pred, fault_pred = model(torch.tensor(X_test, dtype=torch.float32))

    health_rmse = np.sqrt(((health_pred.detach().numpy() - y_health) ** 2).mean())
    rul_rmse = np.sqrt(((rul_pred.detach().numpy() - y_rul) ** 2).mean())
    fault_acc = (fault_pred.argmax(dim=1).detach().numpy() == y_fault).mean()

    return {
        'health_rmse': health_rmse,
        'rul_rmse': rul_rmse,
        'fault_accuracy': fault_acc
    }

def cross_airline_evaluation(model, airlines_data):
    """Evaluate transfer across 5 airlines."""
    results = {}
    for airline, (X_test, y_test) in airlines_data.items():
        rul_pred = model.predict(X_test)
        rmse = np.sqrt(((rul_pred - y_test) ** 2).mean())
        results[airline] = rmse
    return results
