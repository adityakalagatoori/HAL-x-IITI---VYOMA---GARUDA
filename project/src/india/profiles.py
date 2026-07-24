"""
Airline-Specific Customization Profiles

Each Indian airline has different priorities. GARUDA adapts:
- HAL: Turbojet focus, cost optimization
- Air India: Fleet scale, compliance
- IndiGo: Real-time optimization, high volume
- SpiceJet: Budget consciousness, aging fleet
- Regional carriers: Simplicity, reliability

This is what makes GARUDA "India-first" instead of generic.
"""
from typing import Dict, List
from enum import Enum


class AirlineProfile(Enum):
    """Supported airline profiles."""
    HAL = "hal"
    AIR_INDIA = "air_india"
    INDIGO = "indigo"
    SPICEJET = "spicejet"
    ALLIANCE_AIR = "alliance_air"
    REGIONAL = "regional"


class ProfileConfig:
    """Configuration for airline-specific behavior."""

    PROFILES = {
        AirlineProfile.HAL: {
            'name': 'Hindustan Aeronautics Limited',
            'aircraft_types': ['Turbojet-4S'],  # 4-stage turbojet
            'fleet_size': 50,
            'priorities': ['cost_optimization', 'turbojet_expertise', 'test_bench_validation'],
            'budget_model': 'high_capex_low_opex',  # Can spend on premium solution
            'language': 'English_Hindi',
            'features': {
                'cost_calculator': True,
                'dgca_compliance': True,
                'test_bench_integration': True,
                'supply_chain_optimization': True,
                'maintenance_scheduling': True,
                'crew_interface': False,  # Military aircraft, pilots have training
                'financial_dashboard': True,
                'spare_parts_forecast': True,
                'export_to_excel': True
            },
            'reporting': {
                'frequency': 'Real-time',
                'format': 'DGCA MR format + internal metrics',
                'audit_trail': 'Mandatory',
                'signature': 'Digital signature (SHA-256)'
            },
            'data_sources': {
                'test_bench_data': True,
                'flight_data_recorder': True,
                'maintenance_logs': True,
                'sensor_telemetry': True
            },
            'customization': {
                'health_thresholds': 'Customizable',
                'alert_levels': 'Customizable',
                'maintenance_intervals': 'Customizable',
                'cost_models': 'Customizable (Indian MRO rates)'
            }
        },

        AirlineProfile.AIR_INDIA: {
            'name': 'Air India Limited',
            'aircraft_types': ['Turbofan (Boeing 777, Airbus A350)', 'Turbojet (legacy)'],
            'fleet_size': 150,
            'priorities': ['fleet_optimization', 'regulatory_compliance', 'cost_reduction'],
            'budget_model': 'balanced',
            'language': 'English',
            'features': {
                'cost_calculator': True,
                'dgca_compliance': True,
                'test_bench_integration': False,
                'supply_chain_optimization': True,
                'maintenance_scheduling': True,
                'crew_interface': True,
                'financial_dashboard': True,
                'spare_parts_forecast': True,
                'fleet_wide_view': True,
                'export_to_excel': True
            },
            'reporting': {
                'frequency': 'Daily + on-demand',
                'format': 'DGCA + internal + shareholder reports',
                'audit_trail': 'Mandatory',
                'signature': 'Digital signature + management approval'
            },
            'data_sources': {
                'test_bench_data': False,
                'flight_data_recorder': True,
                'maintenance_logs': True,
                'sensor_telemetry': True,
                'fuel_burn_data': True
            },
            'customization': {
                'health_thresholds': 'Customizable',
                'alert_levels': 'Multi-level (strategic/tactical/operational)',
                'maintenance_intervals': 'Contractual (OEM)',
                'cost_models': 'Indian + international MRO rates'
            }
        },

        AirlineProfile.INDIGO: {
            'name': 'IndiGo Airlines',
            'aircraft_types': ['Turbofan (Airbus A320 family)'],
            'fleet_size': 350,
            'priorities': ['real_time_optimization', 'high_volume_scale', 'cost_per_flight'],
            'budget_model': 'cost_sensitive',
            'language': 'English',
            'features': {
                'cost_calculator': True,
                'dgca_compliance': True,
                'test_bench_integration': False,
                'supply_chain_optimization': True,
                'maintenance_scheduling': True,
                'crew_interface': True,
                'financial_dashboard': True,
                'spare_parts_forecast': True,
                'fleet_wide_view': True,
                'real_time_updates': True,  # IndiGo-specific
                'route_optimization': True,  # IndiGo-specific
                'export_to_excel': True
            },
            'reporting': {
                'frequency': 'Real-time (4x daily)',
                'format': 'Mobile-friendly dashboards',
                'audit_trail': 'Optional (performance metrics)',
                'signature': None
            },
            'data_sources': {
                'test_bench_data': False,
                'flight_data_recorder': True,
                'maintenance_logs': True,
                'sensor_telemetry': True,
                'fuel_burn_data': True,
                'route_data': True,
                'crew_reports': True
            },
            'customization': {
                'health_thresholds': 'Preset (optimized for A320)',
                'alert_levels': 'Automatic (ML-driven)',
                'maintenance_intervals': 'Contractual + predictive',
                'cost_models': 'Indian MRO rates only (budget focus)'
            }
        },

        AirlineProfile.SPICEJET: {
            'name': 'SpiceJet Limited',
            'aircraft_types': ['Turbofan (Boeing 737, ATR)', 'Some older aircraft'],
            'fleet_size': 100,
            'priorities': ['cost_minimization', 'aging_fleet_support', 'downtime_reduction'],
            'budget_model': 'ultra_cost_sensitive',
            'language': 'English_Hindi',
            'features': {
                'cost_calculator': True,
                'dgca_compliance': True,
                'test_bench_integration': False,
                'supply_chain_optimization': True,
                'maintenance_scheduling': True,
                'crew_interface': True,
                'financial_dashboard': True,
                'spare_parts_forecast': False,  # Too expensive, basic only
                'fleet_wide_view': False,  # Simplified UI
                'export_to_excel': False,  # PDF only
                'simple_alerts': True  # SpiceJet-specific (simplified UI)
            },
            'reporting': {
                'frequency': 'Daily summary',
                'format': 'Simple PDF reports',
                'audit_trail': 'Minimal',
                'signature': 'None'
            },
            'data_sources': {
                'test_bench_data': False,
                'flight_data_recorder': False,  # Older aircraft may not have
                'maintenance_logs': True,
                'sensor_telemetry': True
            },
            'customization': {
                'health_thresholds': 'Preset (conservative)',
                'alert_levels': 'Two-level: OK / Needs Service',
                'maintenance_intervals': 'Contractual only',
                'cost_models': 'Indian MRO rates (budget focus) + DIY options'
            }
        },

        AirlineProfile.ALLIANCE_AIR: {
            'name': 'Alliance Air (Air India Regional)',
            'aircraft_types': ['ATR 42/72', 'Some turboprops'],
            'fleet_size': 40,
            'priorities': ['reliability', 'simplicity', 'cost'],
            'budget_model': 'budget_conscious',
            'language': 'English_Hindi',
            'features': {
                'cost_calculator': True,
                'dgca_compliance': True,
                'test_bench_integration': False,
                'supply_chain_optimization': False,
                'maintenance_scheduling': True,
                'crew_interface': True,
                'financial_dashboard': False,
                'spare_parts_forecast': False,
                'fleet_wide_view': False,
                'export_to_excel': False,
                'simple_alerts': True
            },
            'reporting': {
                'frequency': 'Weekly summary',
                'format': 'Simple text reports',
                'audit_trail': 'Basic',
                'signature': 'None'
            },
            'data_sources': {
                'test_bench_data': False,
                'flight_data_recorder': False,
                'maintenance_logs': True,
                'sensor_telemetry': True
            },
            'customization': {
                'health_thresholds': 'Preset (regional aircraft focused)',
                'alert_levels': 'One-level: Schedule Maintenance',
                'maintenance_intervals': 'ATR factory defaults',
                'cost_models': 'Indian MRO rates (regional focus)'
            }
        }
    }

    @staticmethod
    def get_profile(airline: AirlineProfile) -> Dict:
        """Get configuration for specific airline."""
        return ProfileConfig.PROFILES.get(airline)


class AirlineCustomizer:
    """Customizes GARUDA for specific airline."""

    def __init__(self, airline_profile: AirlineProfile):
        self.profile = ProfileConfig.get_profile(airline_profile)
        self.airline_name = self.profile['name']
        self.config = self.profile

    def get_ui_configuration(self) -> Dict:
        """Return UI settings for this airline."""
        return {
            'dashboard_widgets': self._get_dashboard_widgets(),
            'alert_system': self._get_alert_system(),
            'reporting_options': self._get_reporting_options(),
            'language': self.config['language'],
            'theme': self._get_theme()
        }

    def _get_dashboard_widgets(self) -> List[str]:
        """Widgets to show on dashboard."""
        widgets = []
        if self.config['features']['financial_dashboard']:
            widgets.extend(['cost_savings', 'roi_calculator', 'budget_tracker'])
        if self.config['features']['fleet_wide_view']:
            widgets.extend(['fleet_health_summary', 'aircraft_status_map', 'bulk_actions'])
        if self.config['features']['real_time_updates']:
            widgets.append('live_feed')
        if self.config['features']['simple_alerts']:
            widgets.append('simplified_alerts')
        else:
            widgets.append('detailed_alerts')
        return widgets

    def _get_alert_system(self) -> Dict:
        """Alert configuration."""
        levels = self.config['customization']['alert_levels']

        if levels == 'One-level: Schedule Maintenance':
            return {
                'levels': 1,
                'format': 'Simple: "Schedule Maintenance by cycle X"'
            }
        elif levels == 'Two-level: OK / Needs Service':
            return {
                'levels': 2,
                'format': 'OK / Needs Service'
            }
        else:
            return {
                'levels': 3,
                'format': 'Green / Yellow / Red (with sub-levels)'
            }

    def _get_reporting_options(self) -> List[str]:
        """Available report types."""
        options = []
        if self.config['features']['dgca_compliance']:
            options.append('DGCA MR Format')
        if self.config['features']['financial_dashboard']:
            options.extend(['Cost Summary', 'ROI Report'])
        if self.config['features']['export_to_excel']:
            options.append('Excel Export')
        elif not self.config['features']['export_to_excel']:
            options.append('PDF Only')
        return options

    def _get_theme(self) -> Dict:
        """UI theme preferences."""
        # Budget airlines get simpler themes
        if 'budget' in self.config['budget_model']:
            return {
                'colors': 'Simple (green/red)',
                'complexity': 'Low',
                'responsive': 'Mobile-first'
            }
        else:
            return {
                'colors': 'Professional (multi-color)',
                'complexity': 'High',
                'responsive': 'Desktop-first with mobile support'
            }

    def get_cost_model(self) -> Dict:
        """Get cost model for this airline."""
        models = {
            'high_capex_low_opex': 'Premium solution costs justified by savings',
            'balanced': 'Balanced cost-benefit calculation',
            'cost_sensitive': 'Focus on direct cost reductions',
            'ultra_cost_sensitive': 'Minimal features, maximum savings visualization',
            'budget_conscious': 'Basic features, essential metrics only'
        }

        return {
            'model_type': self.config['budget_model'],
            'description': models.get(self.config['budget_model']),
            'garuda_pricing': self._get_pricing_tier(),
            'expected_roi': self._get_roi_expectation()
        }

    def _get_pricing_tier(self) -> str:
        """Pricing tier for this airline."""
        pricing = {
            'high_capex_low_opex': '₹3Cr/year (Premium)',
            'balanced': '₹2Cr/year (Professional)',
            'cost_sensitive': '₹1.5Cr/year (Standard)',
            'ultra_cost_sensitive': '₹1Cr/year (Basic)',
            'budget_conscious': '₹80L/year (Essential)'
        }
        return pricing.get(self.config['budget_model'])

    def _get_roi_expectation(self) -> str:
        """Expected ROI for this airline type."""
        fleet_size = self.config['fleet_size']
        model = self.config['budget_model']

        if model == 'high_capex_low_opex':
            return f"₹50Cr+/year savings (fleet: {fleet_size})"
        elif model == 'balanced':
            return f"₹25-30Cr/year savings (fleet: {fleet_size})"
        elif model == 'cost_sensitive':
            return f"₹8-10Cr/year savings (fleet: {fleet_size})"
        elif model == 'ultra_cost_sensitive':
            return f"₹3-5Cr/year savings (fleet: {fleet_size})"
        else:
            return f"₹1-2Cr/year savings (fleet: {fleet_size})"

    def generate_profile_report(self) -> str:
        """Generate customization report."""
        report = []
        report.append("=" * 80)
        report.append(f"GARUDA CUSTOMIZATION PROFILE: {self.airline_name}")
        report.append("=" * 80)

        report.append(f"\nFleet Size: {self.config['fleet_size']} aircraft")
        report.append(f"Aircraft Types: {', '.join(self.config['aircraft_types'])}")
        report.append(f"Priorities: {', '.join(self.config['priorities'])}")

        report.append(f"\n--- ENABLED FEATURES ---")
        for feature, enabled in self.config['features'].items():
            status = "✓" if enabled else "✗"
            report.append(f"{status} {feature.replace('_', ' ').title()}")

        ui_config = self.get_ui_configuration()
        report.append(f"\n--- UI CONFIGURATION ---")
        report.append(f"Language: {ui_config['language']}")
        report.append(f"Dashboard Widgets: {len(ui_config['dashboard_widgets'])}")
        for widget in ui_config['dashboard_widgets']:
            report.append(f"  • {widget}")

        cost_model = self.get_cost_model()
        report.append(f"\n--- FINANCIAL MODEL ---")
        report.append(f"Budget Model: {cost_model['model_type']}")
        report.append(f"GARUDA Pricing: {cost_model['garuda_pricing']}")
        report.append(f"Expected ROI: {cost_model['expected_roi']}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


if __name__ == "__main__":
    print("Airline-Specific Profiles module ready")
    print("Customizes GARUDA for each Indian carrier\n")

    # Example: IndiGo customization
    customizer = AirlineCustomizer(AirlineProfile.INDIGO)
    print(customizer.generate_profile_report())
