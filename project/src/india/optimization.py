"""
Maintenance Cost Optimizer

Critical for Indian airlines: Quantifies ₹ savings from predictive maintenance.
Makes ROI visible: "GARUDA saves ₹22 crores/year" vs competitor's "Predicts health".
"""
import numpy as np
from typing import Dict, List
import pandas as pd


class IndianMROCosts:
    """Indian Maintenance Repair Overhaul cost models."""

    # Indian MRO costs (₹ estimates based on industry averages)
    COSTS = {
        'compressor_fouling_wash': 50_000,  # ₹50K
        'compressor_blade_replacement': 25_00_000,  # ₹25L
        'combustor_erosion_repair': 15_00_000,  # ₹15L
        'turbine_blade_replacement': 35_00_000,  # ₹35L
        'full_engine_overhaul': 2_50_00_000,  # ₹2.5Cr
        'unscheduled_removal': 5_00_00_000,  # ₹5Cr
        'spare_aircraft_rental': 10_00_000,  # ₹10L per day
        'revenue_loss_per_day_grounded': 1_00_00_000,  # ₹1Cr per day
        'crew_repositioning': 5_00_000,  # ₹5L per flight
        'maintenance_labor_per_hour': 5_000,  # ₹5K per technician-hour
        'emergency_parts_premium': 0.3,  # 30% premium for express delivery
        'fuel_consumption_per_cycle': 1_000,  # ₹1K per cycle (degraded engine)
    }

    @staticmethod
    def get_maintenance_cost(component: str) -> int:
        """Get cost for specific maintenance action."""
        return IndianMROCosts.COSTS.get(component, 0)

    @staticmethod
    def calculate_maintenance_hours(component: str) -> int:
        """Estimate labor hours for maintenance (varies by component)."""
        hours_map = {
            'compressor_fouling_wash': 4,
            'compressor_blade_replacement': 80,
            'combustor_erosion_repair': 60,
            'turbine_blade_replacement': 120,
            'full_engine_overhaul': 800,
        }
        return hours_map.get(component, 0)


class MaintenanceCostCalculator:
    """Calculates costs for predictive vs reactive maintenance."""

    def __init__(self, aircraft_utilization_hours_per_year: int = 2500):
        """
        Args:
            aircraft_utilization_hours_per_year: Typical 2500-3000 hrs/year for Indian carriers
        """
        self.util_hours = aircraft_utilization_hours_per_year

    def cost_of_scheduled_maintenance(self, component: str, labor_hours: int = None) -> Dict:
        """Cost of planned maintenance (occurs at scheduled intervals)."""
        parts_cost = IndianMROCosts.get_maintenance_cost(component)
        labor_hours = labor_hours or IndianMROCosts.calculate_maintenance_hours(component)
        labor_cost = labor_hours * IndianMROCosts.get_maintenance_cost('maintenance_labor_per_hour')

        total = parts_cost + labor_cost

        return {
            'component': component,
            'parts_cost': parts_cost,
            'labor_hours': labor_hours,
            'labor_cost': labor_cost,
            'total_cost': total,
            'type': 'SCHEDULED'
        }

    def cost_of_unscheduled_failure(self, component: str, cycles_between_failure: int) -> Dict:
        """Cost when failure occurs unexpectedly (catastrophic)."""
        # Unscheduled failure multiplier: things cost more when emergency
        failure_cost = IndianMROCosts.get_maintenance_cost('unscheduled_removal')
        replacement_parts = IndianMROCosts.get_maintenance_cost(component) * 2  # 2x cost for emergency parts
        downtime_days = 7  # Typical: 1 week to get parts, repair, test
        downtime_cost = IndianMROCosts.get_maintenance_cost('revenue_loss_per_day_grounded') * downtime_days
        spare_aircraft = IndianMROCosts.get_maintenance_cost('spare_aircraft_rental') * downtime_days
        crew_cost = IndianMROCosts.get_maintenance_cost('crew_repositioning')

        total = failure_cost + replacement_parts + downtime_cost + spare_aircraft + crew_cost

        return {
            'component': component,
            'failure_removal_cost': failure_cost,
            'parts_cost_emergency': replacement_parts,
            'downtime_days': downtime_days,
            'revenue_loss': downtime_cost,
            'spare_aircraft_cost': spare_aircraft,
            'crew_repositioning_cost': crew_cost,
            'total_cost': total,
            'type': 'UNSCHEDULED',
            'severity': 'CATASTROPHIC'
        }

    def cost_of_health_degradation(self, cycles_predicted: int, health_loss: float) -> Dict:
        """Fuel cost impact from engine degradation."""
        # Degraded engine burns 5-10% more fuel
        extra_fuel_consumption = health_loss * 0.08  # 8% per unit health loss

        # Rough estimate: ₹1K extra per cycle with degraded engine
        fuel_cost_per_cycle = IndianMROCosts.get_maintenance_cost('fuel_consumption_per_cycle')
        total_extra_fuel_cost = extra_fuel_consumption * cycles_predicted * fuel_cost_per_cycle

        return {
            'health_loss': health_loss,
            'extra_fuel_consumption_pct': extra_fuel_consumption * 100,
            'cycles': cycles_predicted,
            'cost_per_cycle': fuel_cost_per_cycle,
            'total_extra_fuel_cost': total_extra_fuel_cost,
            'type': 'OPERATIONAL'
        }


class PredictiveVsReactiveROI:
    """Compare ROI: Predictive maintenance vs waiting for failure."""

    def __init__(self, aircraft_fleet_size: int = 20):
        """
        Args:
            aircraft_fleet_size: Number of aircraft in airline's fleet
        """
        self.fleet_size = aircraft_fleet_size
        self.calc = MaintenanceCostCalculator()

    def roi_scenario(self, component: str, annual_failure_rate: float = 0.15) -> Dict:
        """
        Compare costs over 1 year.

        Args:
            component: Engine component
            annual_failure_rate: What fraction of fleet has failure per year (typical 10-20%)
        """
        # SCENARIO 1: Reactive (wait until failure)
        failures_per_year = self.fleet_size * annual_failure_rate
        reactive_cost = failures_per_year * self.calc.cost_of_unscheduled_failure(
            component, cycles_between_failure=1000)['total_cost']

        # SCENARIO 2: Predictive (maintain based on health)
        predictive_cost = self.fleet_size * self.calc.cost_of_scheduled_maintenance(component)['total_cost']

        # Savings
        savings = reactive_cost - predictive_cost
        roi_percent = (savings / predictive_cost * 100) if predictive_cost > 0 else 0

        return {
            'component': component,
            'fleet_size': self.fleet_size,
            'annual_failure_rate': annual_failure_rate,
            'reactive_annual_cost': reactive_cost,
            'predictive_annual_cost': predictive_cost,
            'annual_savings': savings,
            'roi_percent': roi_percent,
            'payback_months': (predictive_cost / (savings / 12)) if savings > 0 else float('inf')
        }

    def fleet_wide_roi(self) -> Dict:
        """Calculate total ROI across entire fleet."""
        components = ['compressor_fouling_wash', 'combustor_erosion_repair', 'turbine_blade_replacement']

        total_reactive = 0
        total_predictive = 0

        scenario_details = []
        for comp in components:
            scenario = self.roi_scenario(comp)
            scenario_details.append(scenario)
            total_reactive += scenario['reactive_annual_cost']
            total_predictive += scenario['predictive_annual_cost']

        total_savings = total_reactive - total_predictive
        total_roi = (total_savings / total_predictive * 100) if total_predictive > 0 else 0

        return {
            'fleet_size': self.fleet_size,
            'components_monitored': len(components),
            'total_reactive_annual_cost': total_reactive,
            'total_predictive_annual_cost': total_predictive,
            'total_annual_savings': total_savings,
            'total_roi_percent': total_roi,
            'payback_months': (total_predictive / (total_savings / 12)) if total_savings > 0 else float('inf'),
            'scenario_details': scenario_details
        }


class AirlineROIDashboard:
    """Generate ROI dashboard for airlines (in ₹)."""

    @staticmethod
    def generate_dashboard(airline_name: str, fleet_size: int = 20) -> str:
        """Generate visual ROI dashboard in text format."""
        roi_calc = PredictiveVsReactiveROI(fleet_size)
        fleet_roi = roi_calc.fleet_wide_roi()

        dashboard = []
        dashboard.append("=" * 80)
        dashboard.append(f"GARUDA ROI DASHBOARD - {airline_name.upper()}")
        dashboard.append("=" * 80)

        dashboard.append(f"\nFleet Size: {fleet_size} aircraft")

        dashboard.append("\n--- ANNUAL COST COMPARISON ---")
        dashboard.append(f"Reactive Maintenance (Current):  ₹{fleet_roi['total_reactive_annual_cost']:,}")
        dashboard.append(f"Predictive Maintenance (GARUDA):₹{fleet_roi['total_predictive_annual_cost']:,}")
        dashboard.append(f"\nAnnual Savings:                  ₹{fleet_roi['total_annual_savings']:,}")
        dashboard.append(f"ROI:                            {fleet_roi['total_roi_percent']:.1f}%")
        dashboard.append(f"Payback Period:                 {fleet_roi['payback_months']:.1f} months")

        dashboard.append("\n--- COMPONENT-WISE BREAKDOWN ---")
        for scenario in fleet_roi['scenario_details']:
            dashboard.append(f"\n{scenario['component'].upper()}")
            dashboard.append(f"  Reactive: ₹{scenario['reactive_annual_cost']:,}")
            dashboard.append(f"  Predictive: ₹{scenario['predictive_annual_cost']:,}")
            dashboard.append(f"  Savings: ₹{scenario['annual_savings']:,} ({scenario['roi_percent']:.0f}%)")

        dashboard.append("\n--- 5-YEAR PROJECTION ---")
        five_year_savings = fleet_roi['total_annual_savings'] * 5
        five_year_garuda_cost = 1_50_00_000 * 5  # Assume ₹1.5Cr/year for GARUDA
        net_five_year = five_year_savings - five_year_garuda_cost

        dashboard.append(f"5-Year Maintenance Savings:     ₹{five_year_savings:,}")
        dashboard.append(f"5-Year GARUDA Subscription:     ₹{five_year_garuda_cost:,}")
        dashboard.append(f"Net 5-Year Benefit:             ₹{net_five_year:,}")

        dashboard.append("\n" + "=" * 80)
        dashboard.append("CONCLUSION: Predictive maintenance through GARUDA is cost-effective")
        dashboard.append("and becomes highly profitable over multi-year operations.")
        dashboard.append("=" * 80)

        return "\n".join(dashboard)


if __name__ == "__main__":
    print("Maintenance Cost Optimizer module ready")
    print("Quantifies ₹ savings for Indian airlines\n")

    # Example: Air India with 50 aircraft
    print(AirlineROIDashboard.generate_dashboard("Air India", fleet_size=50))
