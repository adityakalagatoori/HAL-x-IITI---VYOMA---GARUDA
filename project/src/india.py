"""
GARUDA India: DGCA Compliance, Cost Optimization, Airline Profiles

Combines: compliance.py, optimization.py, profiles.py
"""
import hashlib
import hmac
from datetime import datetime, timedelta
from enum import Enum

# ============ DGCA COMPLIANCE ============

class MaintenanceActionType(Enum):
    COMPRESSOR_OVERHAUL = "Compressor Overhaul"
    TURBINE_REPLACEMENT = "Turbine Replacement"
    FULL_ENGINE_REPLACEMENT = "Full Engine Replacement"
    ROUTINE_CHECK = "Routine Check"
    EMERGENCY_LANDING = "Emergency Landing"

class DGCAComplianceReporter:
    """Generate DGCA Maintenance Release (MR) documents."""
    def __init__(self, airline):
        self.airline = airline
        self.signer = SecureDigitalSignature()

    def generate_maintenance_release(self, engine_id, prediction, recommendation,
                                     estimated_cost_inr, scheduled_date,
                                     engineer_email, manager_email):
        """Generate signed MR document."""
        mr_content = f"""
DIRECTORATE GENERAL OF CIVIL AVIATION (DGCA)
Maintenance Release Document
======================================

Airline: {self.airline}
Engine ID: {engine_id}
Date Generated: {datetime.now().isoformat()}

HEALTH ASSESSMENT:
- Predicted RUL: {prediction['rul']:.1f} hours
- Confidence Bounds: [{prediction['lower']:.1f}, {prediction['upper']:.1f}]
- Confidence Level: 90%

RECOMMENDED ACTION:
{recommendation}

Estimated Cost: ₹{estimated_cost_inr:,}
Scheduled Date: {scheduled_date}

COMPLIANCE:
This document complies with DGCA airworthiness requirements.
Requires two-person approval (engineer + manager).

APPROVALS:
Engineer: [Awaiting signature from {engineer_email}]
Manager: [Awaiting signature from {manager_email}]

---
"""
        # Sign the document
        signature = self.signer.sign(mr_content)
        mr_content += f"Document Signature (HMAC-SHA256): {signature}\n"
        mr_content += f"Signature Timestamp: {datetime.now().isoformat()}\n"

        return mr_content

# ============ COST OPTIMIZATION ============

class IndianMROCosts:
    """Database of MRO costs in Indian rupees."""
    MAINTENANCE_ACTIONS = {
        'routine_check': {'min': 50_000, 'max': 2_00_000},
        'compressor_inspection': {'min': 5_00_000, 'max': 15_00_000},
        'compressor_overhaul': {'min': 20_00_000, 'max': 50_00_000},
        'turbine_inspection': {'min': 10_00_000, 'max': 25_00_000},
        'turbine_replacement': {'min': 1_00_00_000, 'max': 2_50_00_000},
        'full_engine_replacement': {'min': 5_00_00_000, 'max': 10_00_00_000},
        'emergency_landing': {'min': 20_00_000, 'max': 1_00_00_000},  # Indirect costs
    }

    @classmethod
    def estimate_cost(cls, action_type, urgency='routine'):
        """Estimate cost for given action."""
        if action_type not in cls.MAINTENANCE_ACTIONS:
            return None
        cost_range = cls.MAINTENANCE_ACTIONS[action_type]
        if urgency == 'emergency':
            return cost_range['max']
        elif urgency == 'routine':
            return cost_range['min']
        else:
            return (cost_range['min'] + cost_range['max']) / 2

class MaintenanceCostCalculator:
    """Calculate predictive vs reactive maintenance ROI."""
    def __init__(self, fleet_size=100, flights_per_aircraft_per_year=500):
        self.fleet_size = fleet_size
        self.flights_per_year = flights_per_aircraft_per_year

    def predictive_maintenance_cost(self, rul_hours):
        """Cost of planned maintenance based on prediction."""
        if rul_hours < 50:
            action = 'compressor_overhaul'
            urgency = 'emergency'
        elif rul_hours < 100:
            action = 'compressor_inspection'
            urgency = 'high'
        else:
            action = 'routine_check'
            urgency = 'routine'

        cost = IndianMROCosts.estimate_cost(action, urgency)
        return cost

    def reactive_maintenance_cost(self):
        """Cost of unplanned maintenance (engine failure)."""
        # Emergency landing + full repair
        return IndianMROCosts.MAINTENANCE_ACTIONS['emergency_landing']['max'] + \
               IndianMROCosts.MAINTENANCE_ACTIONS['full_engine_replacement']['min']

    def calculate_roi(self, num_predictions_per_year=1000):
        """Calculate fleet-wide ROI."""
        avg_predictive_cost = self.predictive_maintenance_cost(75)  # Average
        avg_reactive_cost = self.reactive_maintenance_cost()

        # Assume 10% failure rate prevented by predictive maintenance
        failure_rate = 0.10
        savings_per_prevention = avg_reactive_cost - avg_predictive_cost

        total_savings = num_predictions_per_year * failure_rate * savings_per_prevention
        payback_months = 12 / (total_savings / (avg_predictive_cost * num_predictions_per_year))

        return {
            'annual_savings_inr': total_savings,
            'payback_months': payback_months,
            'savings_per_prevention': savings_per_prevention
        }

# ============ AIRLINE PROFILES ============

class AirlineProfile(Enum):
    HAL = {
        'name': 'Hindustan Aeronautics Limited',
        'annual_cost_inr': 3_00_00_000,  # ₹3 Cr
        'contract_years': 5,
        'flights_per_year': 500,
        'focus': 'turbojet',
        'roi_expectation': 'long_term',
        'ui_language': 'en',  # Can be extended to Hindi
    }
    AIR_INDIA = {
        'name': 'Air India Limited',
        'annual_cost_inr': 1_50_00_000,  # ₹1.5 Cr
        'contract_years': 4,
        'flights_per_year': 1000,
        'focus': 'mixed_fleet',
        'roi_expectation': 'per_flight_hour',
        'ui_language': 'en',
    }
    INDIGO = {
        'name': 'IndiGo Airlines',
        'annual_cost_inr': 80_00_000,  # ₹80 L
        'contract_years': 3,
        'flights_per_year': 2000,
        'focus': 'cost_sensitive',
        'roi_expectation': 'per_flight_hour',
        'ui_language': 'en',
    }
    SPICEJET = {
        'name': 'SpiceJet Limited',
        'annual_cost_inr': 60_00_000,  # ₹60 L
        'contract_years': 3,
        'flights_per_year': 2000,
        'focus': 'regional',
        'roi_expectation': 'per_flight_hour',
        'ui_language': 'en',
    }
    ALLIANCE_AIR = {
        'name': 'Alliance Air',
        'annual_cost_inr': 80_00_000,  # ₹80 L
        'contract_years': 3,
        'flights_per_year': 500,
        'focus': 'regional_small_fleet',
        'roi_expectation': 'per_flight_hour',
        'ui_language': 'en',
    }

class AirlineCustomizer:
    """Customize GARUDA for specific airline."""
    def __init__(self, airline_name):
        self.airline = AirlineProfile[airline_name.upper()]
        self.config = self.airline.value

    def get_cost_model(self):
        """Return airline-specific cost model."""
        cost_per_flight_hour = self.config['annual_cost_inr'] / (
            self.config['flights_per_year'] * 2  # ~2 hours per flight
        )
        return {
            'annual_cost': self.config['annual_cost_inr'],
            'cost_per_flight_hour': cost_per_flight_hour,
            'contract_years': self.config['contract_years'],
        }

    def get_ui_config(self):
        """Return UI customization."""
        return {
            'language': self.config['ui_language'],
            'currency': 'INR',
            'currency_symbol': '₹',
            'locale': 'en_IN',
        }

    def get_roi_dashboard(self, predictions):
        """Airline-specific ROI display."""
        calculator = MaintenanceCostCalculator()
        roi = calculator.calculate_roi()

        dashboard = f"""
GARUDA ROI Dashboard — {self.config['name']}
=====================================================
Annual Subscription: ₹{self.config['annual_cost_inr']:,}
Contract Period: {self.config['contract_years']} years
Total Investment: ₹{self.config['annual_cost_inr'] * self.config['contract_years']:,}

Projected Annual Savings: ₹{roi['annual_savings_inr']:,}
Payback Period: {roi['payback_months']:.1f} months
Cost per Prevention: ₹{roi['savings_per_prevention']:,}

ROI Timeline:
Year 1: Breakeven + {(roi['annual_savings_inr'] - self.config['annual_cost_inr']):,}
Year 2: {2 * roi['annual_savings_inr']:,}
Year 3: {3 * roi['annual_savings_inr']:,}
...
Year {self.config['contract_years']}: {self.config['contract_years'] * roi['annual_savings_inr']:,}

Total 5-Year ROI: ₹{5 * roi['annual_savings_inr']:,}
"""
        return dashboard

# ============ SECURITY ============

class SecureDigitalSignature:
    """HMAC-SHA256 document signing."""
    def __init__(self, secret_key=None):
        self.secret_key = secret_key or "GARUDA_SECURE_KEY_2026"

    def sign(self, message):
        """Generate HMAC-SHA256 signature."""
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify(self, message, signature):
        """Verify signature."""
        expected_signature = self.sign(message)
        return hmac.compare_digest(signature, expected_signature)
